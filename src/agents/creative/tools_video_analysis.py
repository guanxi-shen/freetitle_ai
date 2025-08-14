"""
Video content analysis tools using Gemini 2.5 Pro
Analyzes video clips for motion, peaks, effectiveness in parallel
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from ...core.llm import get_llm
from ...core.state import RAGState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_video_clips(
    clip_urls: Optional[List[str]] = None,
    state: Optional[RAGState] = None
) -> Dict[str, Any]:
    """
    Analyze video clips for motion, peaks, effectiveness using Gemini 2.5 Pro

    Executes parallel analysis for all clips to minimize latency.
    Results are cached in state for reuse.

    Args:
        clip_urls: List of video URLs to analyze (default: all completed clips from state)
        state: RAGState for accessing video tasks and caching results

    Returns:
        Aggregated analysis JSON:
        {
            "clips": [
                {
                    "filename": "sc01_sh01_video_v1.mp4",
                    "url": "https://...",
                    "duration_seconds": 8.0,
                    "motion_intensity": {
                        "timeline": [0.2, 0.5, 0.8, ...],
                        "peak_moments": [3.2, 4.8],
                        "static_ranges": [[5.5, 7.2]]
                    },
                    "content_summary": "Character enters...",
                    "first_frame": {"motion": "static", "description": "..."},
                    "last_frame": {"motion": "fast_pan_right", "description": "..."},
                    "trim_recommendation": {
                        "suggested_start": 0.8,
                        "suggested_end": 7.4,
                        "reason": "..."
                    }
                }
            ],
            "analysis_timestamp": "ISO timestamp",
            "total_clips": 5
        }
    """

    logger.info("=" * 80)
    logger.info("[Video Analysis] Starting video content analysis...")
    logger.info("=" * 80)

    # Collect clips to analyze
    clips_to_analyze = []

    if clip_urls:
        # Use provided URLs
        for url in clip_urls:
            # Extract filename from URL
            filename = url.split('/')[-1].split('?')[0]  # Remove query params
            clips_to_analyze.append({
                "filename": filename,
                "url": url
            })
    elif state:
        # Extract from asset_urls (source of truth)
        generated_videos = state.get("asset_urls", {}).get("generated_videos", [])
        for video in generated_videos:
            scene = video.get("scene", 0)
            shot = video.get("shot", 0)
            version = video.get("version", 1)
            video_url = video.get("url")

            if video_url and scene > 0 and shot > 0:
                filename = f"sc{scene:02d}_sh{shot:02d}_video_v{version}.mp4"

                clips_to_analyze.append({
                    "filename": filename,
                    "url": video_url,
                    "scene": scene,
                    "shot": shot,
                    "version": version
                })

        # Sort by scene/shot for ordered output
        clips_to_analyze.sort(key=lambda x: (x.get("scene", 0), x.get("shot", 0)))
    else:
        logger.error("[Video Analysis] No clips provided and no state available")
        return {
            "clips": [],
            "analysis_timestamp": datetime.now().isoformat(),
            "total_clips": 0,
            "error": "No clips to analyze"
        }

    if not clips_to_analyze:
        logger.warning("[Video Analysis] No completed clips found")
        return {
            "clips": [],
            "analysis_timestamp": datetime.now().isoformat(),
            "total_clips": 0
        }

    logger.info(f"[Video Analysis] Analyzing {len(clips_to_analyze)} clips in parallel...")

    # Check cache in state
    if state and "video_analysis_metadata" in state:
        cached_metadata = state["video_analysis_metadata"]
    else:
        cached_metadata = {}

    # Parallel analysis with ThreadPoolExecutor
    analyzed_clips = []
    clips_needing_analysis = []

    for clip in clips_to_analyze:
        if clip["filename"] in cached_metadata:
            # Use cached result
            cached_result = cached_metadata[clip["filename"]]
            analyzed_clips.append({
                "filename": clip["filename"],
                "url": clip["url"],
                **cached_result
            })
            logger.info(f"[Video Analysis] Using cached analysis for {clip['filename']}")
        else:
            clips_needing_analysis.append(clip)

    # Analyze uncached clips in parallel
    if clips_needing_analysis:
        with ThreadPoolExecutor(max_workers=min(12, len(clips_needing_analysis))) as executor:
            # Submit all analysis tasks
            future_to_clip = {
                executor.submit(_analyze_single_clip, clip["url"], clip["filename"]): clip
                for clip in clips_needing_analysis
            }

            # Collect results as they complete
            for future in as_completed(future_to_clip):
                clip = future_to_clip[future]
                try:
                    result = future.result()
                    analyzed_clips.append({
                        "filename": clip["filename"],
                        "url": clip["url"],
                        **result
                    })

                    # Cache in state
                    if state:
                        if "video_analysis_metadata" not in state:
                            state["video_analysis_metadata"] = {}
                        state["video_analysis_metadata"][clip["filename"]] = result

                    logger.info(f"[Video Analysis] Completed analysis for {clip['filename']}")
                except Exception as e:
                    logger.error(f"[Video Analysis] Error analyzing {clip['filename']}: {str(e)}")
                    # Add error placeholder
                    analyzed_clips.append({
                        "filename": clip["filename"],
                        "url": clip["url"],
                        "analysis_failed": True,
                        "error": str(e)
                    })

    # Sort final results by original order
    if clips_to_analyze[0].get("scene") is not None:
        analyzed_clips.sort(key=lambda x: (
            next((c.get("scene", 0) for c in clips_to_analyze if c["filename"] == x["filename"]), 0),
            next((c.get("shot", 0) for c in clips_to_analyze if c["filename"] == x["filename"]), 0)
        ))

    # Build aggregated response
    aggregated_result = {
        "clips": analyzed_clips,
        "analysis_timestamp": datetime.now().isoformat(),
        "total_clips": len(analyzed_clips)
    }

    success_count = sum(1 for c in analyzed_clips if not c.get("analysis_failed"))
    failed_count = sum(1 for c in analyzed_clips if c.get("analysis_failed"))

    logger.info("=" * 80)
    logger.info(f"[Video Analysis] Analysis complete!")
    logger.info(f"[Video Analysis] Total clips: {len(analyzed_clips)}")
    logger.info(f"[Video Analysis] Successful: {success_count}")
    logger.info(f"[Video Analysis] Failed: {failed_count}")
    logger.info("=" * 80)

    return aggregated_result


def _analyze_single_clip(video_url: str, filename: str) -> Dict[str, Any]:
    """
    Analyze a single video clip using Gemini 2.5 Pro

    Args:
        video_url: Video URL (GCS signed URL)
        filename: Clip filename for logging

    Returns:
        Analysis result dict (without filename/url, those are added by caller)
    """

    logger.info(f"[Video Analysis] → Analyzing {filename}...")
    logger.debug(f"[Video Analysis]   URL: {video_url[:100]}...")

    # Convert signed URL back to gs:// URI for Gemini
    # Gemini needs gs:// format for from_uri()
    if video_url.startswith("https://storage.googleapis.com/"):
        # Extract bucket and path from signed URL
        # Format: https://storage.googleapis.com/bucket-name/path/to/file.mp4?X-Goog-Algorithm=...
        url_without_params = video_url.split('?')[0]
        path_part = url_without_params.replace("https://storage.googleapis.com/", "")
        gs_uri = f"gs://{path_part}"
        logger.debug(f"[Video Analysis] Converted signed URL to GCS URI: {gs_uri}")
    elif video_url.startswith("gs://"):
        gs_uri = video_url
    else:
        # Unknown format - try as-is
        gs_uri = video_url
        logger.warning(f"[Video Analysis] Unknown URL format: {video_url[:100]}")

    # Prepare video as Part object for Gemini with FPS control
    from google.genai import types

    video_part = types.Part.from_uri(
        file_uri=gs_uri,
        mime_type="video/mp4"
    )

    # Add video metadata to control FPS sampling (12 frames per second for high detail)
    video_part.video_metadata = types.VideoMetadata(fps=12)

    # Analysis prompt
    prompt = """Analyze this AI-generated video clip and provide detailed content analysis for video editing.

Provide the following analysis:

1. **Duration**: Clip duration in seconds

2. **Visual Timeline**: Break down what happens visually with start/end timestamps
   - Format: [{{"start": 0.0, "end": 1.2, "description": "what happens"}}, ...]
   - Focus on major visual events, not every second
   - FLAG any critical AI generation issues in descriptions (morphing, glitches, warping, stuttering, corrupted frames, anatomical/physical inconsistencies)
   - FLAG abrupt end-frame transitions in last 1-2 seconds (dual-frame AI videos may snap to end frame rather than transition progressively)

3. **Audio Timeline**: Break down audio events with start/end timestamps
   - Format: [{{"start": 0.0, "end": 2.0, "description": "ambient sounds, footsteps, dialogue, etc"}}, ...]
   - Include ALL audio in each segment: dialogue, voiceover, singing, music (type, tempo, energy, beat), sound effects, ambient sounds
   - For music: specify genre/type (electronic, orchestral, pop, hip-hop, etc.), tempo (slow/medium/fast), energy (low/medium/high), beat strength (subtle/moderate/strong)
   - Flag audio quality issues: distorted voice, robotic voice, broken/glitchy audio, static, clicking, audio artifacts
   - Examples: "Dialogue: 'Hello' (distorted voice), dramatic orchestral music starts (slow tempo, high energy, strong beat)", "Voiceover: 'In a world...' (clear), epic music continues (building intensity)", "Singing: 'La la la' (clear female voice), upbeat pop music (fast tempo, high energy, steady beat)", "Footsteps, ambient city sounds, soft jazz music (medium tempo, low energy, subtle beat)"

4. **Visual Style**: Describe cinematography, color palette, lighting, camera movement, framing, aesthetic (1-2 sentences)

5. **Opening Sequence**: First few frames/seconds
   - Duration of opening
   - Motion state: "static", "slow_motion", "fast_motion", "pan_left", "pan_right", "pan_up", "pan_down", "zoom_in", "zoom_out"
   - Description of what's visible
   - Subject position: "none", "left", "center", "right"

6. **Closing Sequence**: Last few frames/seconds
   - Duration of closing
   - Motion state (same options as opening)
   - Description of what's visible
   - Subject position: "none", "left", "center", "right"

7. **Peak Moments**: Timestamps of high-value moments
   - Format: [{{"timestamp": 2.8, "type": "action", "description": "flip apex"}}, ...]
   - Types: "action", "emotional", "visual", "audio"

8. **Core Content Ranges**: Segments with highest informational/narrative value
   - Format: [{{"start": 0.7, "end": 4.8, "reason": "main action sequence", "density": "high"}}, ...]
   - Density: "high", "medium", "low"
   - Think: skateboard jump vs preparation, flip vs empty frames

9. **Content Summary**: Brief description (1-2 sentences)

Return your analysis as valid JSON following this exact schema:

{{
  "duration_seconds": 6.1,
  "visual_timeline": [
    {{"start": 0.0, "end": 0.7, "description": "Empty urban street, static shot"}},
    {{"start": 0.7, "end": 2.5, "description": "Character enters left, running"}},
    {{"start": 2.5, "end": 3.8, "description": "Character performs acrobatic flip [GLITCH: product morphs at 3.2s]"}},
    {{"start": 3.8, "end": 4.8, "description": "Character lands, exits right"}},
    {{"start": 4.8, "end": 6.1, "description": "Empty street, static"}}
  ],
  "audio_timeline": [
    {{"start": 0.0, "end": 1.5, "description": "Ambient city sounds (low volume), upbeat electronic music starts (fast tempo, high energy, strong beat)"}},
    {{"start": 1.5, "end": 4.5, "description": "Footsteps, movement sounds, impact, electronic music continues (building intensity, steady beat)"}},
    {{"start": 4.5, "end": 6.1, "description": "Ambient fade out, music ends (gradual fade)"}}
  ],
  "visual_style": "Cinematic nighttime urban aesthetic with cool blue tones and neon accents. Wide angle static camera with wet pavement reflections creating depth.",
  "opening_sequence": {{
    "duration": 0.7,
    "motion": "static",
    "description": "Empty wet city street at night, wide establishing shot",
    "subject_position": "none"
  }},
  "closing_sequence": {{
    "duration": 1.3,
    "motion": "static",
    "description": "Empty street after character exits, same framing",
    "subject_position": "none"
  }},
  "peak_moments": [
    {{"timestamp": 2.8, "type": "action", "description": "Character at flip apex"}},
    {{"timestamp": 3.5, "type": "action", "description": "Landing impact"}}
  ],
  "core_content_ranges": [
    {{
      "start": 0.7,
      "end": 4.8,
      "reason": "Character action sequence with highest narrative value",
      "density": "high"
    }}
  ],
  "content_summary": "Character in futuristic suit runs across night street and performs acrobatic flip"
}}

IMPORTANT: Return ONLY the JSON object, no markdown formatting, no explanations."""

    try:
        # For video analysis, we need to use Google GenAI client directly
        # because videos must be passed as Part.from_uri(), not through the LLM wrapper
        from google import genai
        from ...core.config import PROJECT_ID, CREDENTIALS

        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location="global",  # Use global for multimodal
            credentials=CREDENTIALS
        )

        # Build content with video and prompt
        contents = [video_part, types.Part.from_text(text=prompt)]

        # Generate analysis with Gemini 3 Pro
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=8000,  # Increased for detailed JSON response
                thinking_config=types.ThinkingConfig(
                    thinking_level="high",
                    include_thoughts=True
                ),
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE"
                    )
                ]
            )
        )

        # Extract thinking and text from response (following pattern from llm.py:810-814)
        thinking_text = ""
        response_text = ""

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'thought') and part.thought:
                    thinking_text = part.text or ""
                elif hasattr(part, 'text') and part.text:
                    response_text = part.text

        if not response_text:
            raise ValueError("No response content from Gemini")

        # Format as JSON response matching LLM wrapper pattern (for consistency)
        formatted_response = {
            "content": [
                {"thinking": thinking_text},
                {"text": response_text}
            ]
        }

        # Extract text content (following pattern from agent_script.py:100)
        actual_text = formatted_response['content'][1]['text']

        # Clean JSON using utility function
        from ..base import clean_json_response
        cleaned = clean_json_response(actual_text)

        # Parse JSON
        analysis = json.loads(cleaned)

        # Add analysis timestamp
        analysis["analyzed_at"] = datetime.now().isoformat()

        logger.info(f"[Video Analysis] ✓ {filename}: {analysis.get('duration_seconds', 0):.1f}s")

        return analysis

    except json.JSONDecodeError as e:
        logger.error(f"[Video Analysis] JSON parse error for {filename}: {str(e)}")
        logger.error(f"[Video Analysis] Raw response: {response_text[:500] if 'response_text' in locals() else 'No response'}")
        raise Exception(f"Failed to parse analysis JSON: {str(e)}")

    except Exception as e:
        error_msg = str(e).lower()

        # Detect rate limit errors
        if "429" in error_msg or "rate limit" in error_msg or "quota exceeded" in error_msg or "resource exhausted" in error_msg:
            logger.error("=" * 80)
            logger.error(f"[Video Analysis] RATE LIMIT ERROR detected for {filename}")
            logger.error(f"[Video Analysis] Worker count: 12 workers (may need reduction)")
            logger.error(f"[Video Analysis] Error details: {str(e)}")
            logger.error(f"[Video Analysis] Timestamp: {datetime.now().isoformat()}")
            logger.error(f"[Video Analysis] Recommendation: Consider reducing max_workers to 8-10")
            logger.error("=" * 80)
        else:
            logger.error(f"[Video Analysis] Analysis error for {filename}: {str(e)}")

        raise
