"""
Video editing tools using ffmpeg-python
Simple, manageable functions for the video editor agent
"""

import os
import sys
import json
import logging
import ffmpeg
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set FFmpeg path for Windows if using Chocolatey
if sys.platform == "win32":
    choco_ffmpeg = r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"
    if os.path.exists(choco_ffmpeg):
        # Configure ffmpeg-python to use this path
        os.environ['PATH'] = r"C:\ProgramData\chocolatey\bin;" + os.environ.get('PATH', '')


def prepare_media_for_ffmpeg(media_ref: str) -> str:
    """
    Prepare media reference for FFmpeg (converts GCS URLs to signed URLs)
    FFmpeg supports HTTPS URLs directly but not gs:// protocol

    Args:
        media_ref: GCS URL (gs:// or https://) or local file path

    Returns:
        Signed HTTPS URL for GCS assets, or original path for local files
    """
    # Convert gs:// to signed HTTPS URL (fast, no download)
    if media_ref.startswith("gs://"):
        from ...storage.gcs_utils import generate_signed_url
        signed_url = generate_signed_url(media_ref, expiration_days=1)
        if signed_url:
            return signed_url
        else:
            raise ValueError(f"Error generating signed URL for: {media_ref}")

    # Already HTTPS URL or local path - pass through
    return media_ref


def extract_filename_from_url(url: str) -> str:
    """Extract clean filename from URL (removes path and query params)"""
    return url.split('/')[-1].split('?')[0]


def get_clip_duration(spec: Dict[str, Any], file_path: str) -> float:
    """Get actual clip duration accounting for trim specs"""
    trim = spec.get("trim")
    if trim and "start" in trim and "end" in trim:
        duration = trim["end"] - trim["start"]
        if duration > 0:
            return duration

    # No valid trim - probe for actual duration
    probe = ffmpeg.probe(file_path)
    return float(probe['format']['duration'])


def _probe_single_video(spec: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Probe a single video for metadata (audio presence, duration)
    Designed for parallel execution via ThreadPoolExecutor

    Args:
        spec: Video specification dict with url, filename, trim
        index: Video index for logging

    Returns:
        Dict with probe results and prepared path
    """
    try:
        video_url = spec["url"]
        trim_spec = spec.get("trim")

        # Convert GCS URLs to signed URLs
        prepared_path = prepare_media_for_ffmpeg(video_url)

        # Probe video for metadata
        probe = ffmpeg.probe(prepared_path)
        has_audio = any(s.get('codec_type') == 'audio' for s in probe.get('streams', []))

        # Calculate duration (respects trim spec)
        duration = get_clip_duration(spec, prepared_path)

        logger.debug(f"[Video Editor] Probed video {index+1}: {spec['filename']} - audio={has_audio}, duration={duration:.1f}s")

        return {
            "spec": spec,
            "index": index,
            "prepared_path": prepared_path,
            "has_audio": has_audio,
            "duration": duration,
            "probe": probe,
            "error": None
        }
    except Exception as e:
        logger.error(f"[Video Editor] Probe error for video {index+1} ({spec.get('filename', 'unknown')}): {e}")
        return {
            "spec": spec,
            "index": index,
            "prepared_path": None,
            "has_audio": False,
            "duration": 0,
            "probe": None,
            "error": str(e)
        }


def combine_videos_with_transitions(
    video_specs: List[Dict[str, Any]],
    transitions: Optional[Dict[Tuple[Optional[str], Optional[str]], str]] = None,
    transition_durations: Optional[Dict[Tuple[Optional[str], Optional[str]], float]] = None,
    output_path: Optional[str] = None,
    session_id: str = "",
    aspect_ratio: str = "vertical",
    process_audio: bool = True
) -> Dict:
    """
    Combine videos in order with optional transitions and inline trimming using FFmpeg

    Args:
        video_specs: Ordered list of video specs with inline trim specifications
            Each spec: {"url": str, "filename": str, "trim": {"start": float, "end": float} | None, "mute_audio": bool | None (optional)}
        transitions: Optional dict mapping (from_video, to_video) tuples to transition type
            (None, "sc1_sh1.mp4"): "fade_in" - fade in at start
            ("sc1_sh1.mp4", "sc1_sh2.mp4"): "fade" - clean cross-fade between shots
            ("sc2_sh1.mp4", None): "fade_out" - fade out at end
            Types: fade, fadeslow, fadeblack, fadewhite, hblur, coverleft, coverright, revealleft, revealright, zoomin, squeezeh, squeezev, dissolve
        transition_durations: Optional dict with same keys as transitions, values in seconds (0.2-0.8s range, default 0.5)
        output_path: Optional output path (if None, returns BytesIO buffer)
        session_id: Session ID for GCS operations
        aspect_ratio: Target aspect ratio (vertical/horizontal/square) for normalization
        process_audio: Whether to process audio streams (default True for backwards compatibility)

    Returns:
        Dict with status and output_path (file path) or output_buffer (BytesIO)
    """

    logger.info("=" * 80)
    logger.info(f"[Video Editor] Starting video combination")
    logger.info(f"[Video Editor] Input videos: {len(video_specs)}")
    logger.info(f"[Video Editor] Aspect ratio: {aspect_ratio}")
    if transitions:
        logger.info(f"[Video Editor] Transitions provided: {len(transitions)} total")
        for (from_v, to_v), trans_type in transitions.items():
            logger.info(f"[Video Editor]   Transition key: ({from_v}, {to_v}) -> {trans_type}")
    else:
        logger.info(f"[Video Editor] No transitions provided (will use hard cuts)")
    logger.info("=" * 80)

    if not video_specs:
        logger.error("[Video Editor] No videos provided")
        return {"status": "error", "message": "No videos provided"}

    # Extract filenames from specs for transition matching
    video_filenames = [spec["filename"] for spec in video_specs]

    for i, spec in enumerate(video_specs):
        url = spec["url"]
        filename = spec["filename"]
        trim = spec.get("trim")

        if not url.startswith(("gs://", "https://")) and not os.path.exists(url):
            logger.error(f"[Video Editor] Video {i+1} not found: {url}")
            return {"status": "error", "message": f"Video not found: {url}"}

        if trim:
            logger.info(f"[Video Editor] Video {i+1}: {filename} (trim {trim['start']}s-{trim['end']}s)")
        else:
            logger.info(f"[Video Editor] Video {i+1}: {filename} (full clip)")

    # Determine output mode: file path or memory buffer
    use_buffer = output_path is None
    if use_buffer:
        from io import BytesIO
        output_buffer = BytesIO()
    else:
        output_buffer = None
    
    # Determine target resolution based on aspect ratio
    if aspect_ratio in ["vertical", "9:16"]:
        target_width, target_height = 1080, 1920  # Portrait
    elif aspect_ratio in ["horizontal", "16:9"]:
        target_width, target_height = 1920, 1080  # Landscape
    elif aspect_ratio in ["square", "1:1"]:
        target_width, target_height = 1080, 1080  # Square
    else:
        # Default to vertical if unknown
        target_width, target_height = 1080, 1920

    logger.info(f"[Video Editor] Target resolution: {target_width}x{target_height}")

    try:
        # Phase 1: Probe all videos in parallel
        logger.info(f"[Video Editor] Probing {len(video_specs)} videos in parallel...")

        probe_results = []
        with ThreadPoolExecutor(max_workers=min(12, len(video_specs))) as executor:
            # Submit all probe tasks
            future_to_index = {
                executor.submit(_probe_single_video, spec, i): i
                for i, spec in enumerate(video_specs)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    probe_results.append(result)
                except Exception as e:
                    logger.error(f"[Video Editor] Unexpected error probing video {index+1}: {e}")
                    probe_results.append({
                        "spec": video_specs[index],
                        "index": index,
                        "prepared_path": None,
                        "has_audio": False,
                        "duration": 0,
                        "probe": None,
                        "error": str(e)
                    })

        # Sort results by original index to maintain order
        probe_results.sort(key=lambda x: x["index"])

        # Check for probe errors
        failed_probes = [r for r in probe_results if r["error"]]
        if failed_probes:
            logger.error(f"[Video Editor] {len(failed_probes)} video(s) error probing")
            for failed in failed_probes:
                logger.error(f"[Video Editor]   {failed['spec']['filename']}: {failed['error']}")
            return {"status": "error", "message": f"{len(failed_probes)} video(s) error probing"}

        logger.info(f"[Video Editor] ✓ All videos probed successfully")

        # Calculate video durations early (used by both fade and xfade logic)
        prepared_paths = []
        video_durations = []
        for result in probe_results:
            prepared_paths.append(result["prepared_path"])
            video_durations.append(result["duration"])

        logger.info(f"[Video Editor] Video durations (after trim): {video_durations}")

        # Detect if any clips have original audio that needs preserving
        # Skip audio processing if all clips are muted (fixes aresample topology error)
        all_clips_muted_or_no_audio = all(
            not result["has_audio"] or spec.get("mute_audio", False)
            for result, spec in zip(probe_results, video_specs)
        )

        if process_audio and all_clips_muted_or_no_audio:
            logger.info(f"[Video Editor] All clips muted or no audio - skipping audio processing during concat")
            process_audio = False

        # Phase 2: Load videos with trimming and ensure ALL have audio
        video_audio_pairs = []

        for result in probe_results:
            spec = result["spec"]
            prepared_path = result["prepared_path"]
            has_audio = result["has_audio"]
            duration = result["duration"]
            trim_spec = spec.get("trim")

            # Load video with input seeking (trims both video and audio together)
            if trim_spec and "start" in trim_spec and "end" in trim_spec:
                start = trim_spec["start"]
                end = trim_spec["end"]
                trim_duration = end - start

                # Validate trim duration is positive
                if trim_duration > 0:
                    stream = ffmpeg.input(prepared_path, ss=start, t=trim_duration)
                    logger.info(f"[Video Editor]   Trimmed via input seeking: {start}s to {end}s (duration={trim_duration}s)")
                else:
                    stream = ffmpeg.input(prepared_path)
                    logger.warning(f"[Video Editor]   Invalid trim duration ({trim_duration}s), using full clip for {spec['filename']}")
            else:
                stream = ffmpeg.input(prepared_path)

            # Handle audio based on process_audio flag
            if process_audio:
                # Check per-clip mute flag (optional field)
                mute_audio = spec.get("mute_audio", False)

                # Ensure ALL videos have audio (add silent if missing or muted)
                if has_audio and not mute_audio:
                    audio_stream = stream.audio
                else:
                    # Generate silent audio matching video duration
                    audio_stream = ffmpeg.input('anullsrc=r=44100:cl=stereo', f='lavfi', t=duration)
                    if mute_audio:
                        logger.info(f"[Video Editor] Video {result['index']+1} audio muted per spec (unwanted music or broken audio)")
                    else:
                        logger.info(f"[Video Editor] Video {result['index']+1} has no audio - added {duration:.1f}s silence")
                video_audio_pairs.append((stream.video, audio_stream))
            else:
                # Video-only mode: store video stream with None for audio
                video_audio_pairs.append((stream.video, None))

        # Phase 3: Normalize all videos uniformly
        normalized_videos = []
        audio_streams = []

        for i, (video_stream, audio_stream) in enumerate(video_audio_pairs):
            # Normalize video to target resolution
            normalized = (
                video_stream
                .filter('scale', target_width, target_height,
                       force_original_aspect_ratio='decrease')
                .filter('pad', target_width, target_height, '(ow-iw)/2', '(oh-ih)/2')
                .filter('setsar', '1')  # Fix SAR to 1:1
                .filter('fps', fps=30)  # Normalize framerate to 30fps for xfade compatibility
                .filter('settb', 'AVTB')  # Normalize video timebase AFTER fps (fps resets timebase)
            )

            normalized_videos.append(normalized)

            # Normalize audio timebase only if processing audio
            if process_audio and audio_stream is not None:
                audio_stream = audio_stream.filter('aresample', **{'async': 1})
                audio_streams.append(audio_stream)
            
        # Helper function to match transition keys flexibly
        def find_transition(from_path, to_path):
            """Find transition matching the given paths (handles full path or filename)"""
            # Normalize empty strings to None for robust transition matching
            from_path = from_path if from_path else None
            to_path = to_path if to_path else None

            if not transitions:
                return None, None

            # Try exact match first
            if (from_path, to_path) in transitions:
                trans_type = transitions[(from_path, to_path)]
                trans_dur = transition_durations.get((from_path, to_path), 0.5) if transition_durations else 0.5
                logger.debug(f"[Video Editor] Transition matched (exact): {trans_type} {trans_dur}s")
                return trans_type, trans_dur

            # Try with just filenames (strip query parameters from URLs)
            from_name = from_path.split('/')[-1].split('?')[0] if from_path else None
            to_name = to_path.split('/')[-1].split('?')[0] if to_path else None
            logger.debug(f"[Video Editor] Trying filename match: ({from_name}, {to_name})")
            if (from_name, to_name) in transitions:
                trans_type = transitions[(from_name, to_name)]
                trans_dur = transition_durations.get((from_name, to_name), 0.5) if transition_durations else 0.5
                logger.info(f"[Video Editor] Transition matched (filename): {from_name} -> {to_name} = {trans_type} ({trans_dur}s)")
                return trans_type, trans_dur

            # Try None for start/end transitions
            if from_path is None and (None, to_path) in transitions:
                trans_type = transitions[(None, to_path)]
                trans_dur = transition_durations.get((None, to_path), 0.5) if transition_durations else 0.5
                return trans_type, trans_dur
            if from_path is None and (None, to_name) in transitions:
                trans_type = transitions[(None, to_name)]
                trans_dur = transition_durations.get((None, to_name), 0.5) if transition_durations else 0.5
                return trans_type, trans_dur
            if to_path is None and (from_path, None) in transitions:
                trans_type = transitions[(from_path, None)]
                trans_dur = transition_durations.get((from_path, None), 0.5) if transition_durations else 0.5
                return trans_type, trans_dur
            if to_path is None and (from_name, None) in transitions:
                trans_type = transitions[(from_name, None)]
                trans_dur = transition_durations.get((from_name, None), 0.5) if transition_durations else 0.5
                return trans_type, trans_dur

            return None, None

        # Check for fade in/out at start/end (applied as simple filters)
        fade_in_duration = 0
        fade_out_duration = 0

        if len(normalized_videos) > 0:
            trans_type, trans_dur = find_transition(None, video_filenames[0])
            if trans_type == "fade_in":
                fade_in_duration = trans_dur
                normalized_videos[0] = normalized_videos[0].split()[0].filter('fade', type='in', duration=trans_dur)

        # Build xfade transitions between clips (filenames already extracted earlier)
        xfade_transitions = []
        for i in range(len(video_filenames) - 1):
            # Use filenames for transition matching
            trans_type, trans_dur = find_transition(video_filenames[i], video_filenames[i + 1])
            if trans_type and trans_type not in ["fade_in", "fade_out"]:
                xfade_transitions.append({
                    "index": i,
                    "type": trans_type,
                    "duration": trans_dur
                })
                logger.info(f"[Video Editor] Added xfade transition #{i}: {video_filenames[i]} -> {video_filenames[i+1]} = {trans_type} ({trans_dur}s)")
            else:
                logger.debug(f"[Video Editor] No transition between clips {i} ({video_filenames[i]}) and {i+1} ({video_filenames[i+1]})")

        logger.info(f"[Video Editor] Built {len(xfade_transitions)} xfade transitions from {len(video_specs)-1} possible")

        # Apply fade_out to last clip (only if no incoming xfade to avoid double-fade)
        if len(normalized_videos) > 0:
            trans_type, trans_dur = find_transition(video_filenames[-1], None)
            if trans_type == "fade_out":
                # Check if there's an incoming xfade transition to the last clip
                has_incoming_xfade = any(t["index"] == len(video_filenames) - 2 for t in xfade_transitions)
                if has_incoming_xfade:
                    logger.info(f"[Video Editor] Skipping fade_out filter on last clip (has incoming xfade) - will use timed fade instead")
                    fade_out_duration = trans_dur
                    # Don't apply fade filter here - it will be handled after xfade chain
                else:
                    fade_out_duration = trans_dur
                    # Calculate fade start time from last clip duration
                    last_clip_duration = video_durations[-1]
                    fade_start_time = max(0, last_clip_duration - trans_dur)  # Prevent negative
                    normalized_videos[-1] = normalized_videos[-1].split()[0].filter(
                        'fade',
                        type='out',
                        start_time=fade_start_time,
                        duration=trans_dur
                    )
                    logger.info(f"[Video Editor] Applied fade_out filter to last clip: starts at {fade_start_time:.2f}s, duration {trans_dur}s")

        # Apply xfade transitions
        if len(normalized_videos) == 1:
            final_video = normalized_videos[0]
            final_audio = audio_streams[0] if process_audio else None
            logger.info("[Video Editor] Single video - no transitions needed")
        elif not xfade_transitions:
            # No xfade transitions - use concatenation
            if process_audio:
                # Interleave video and audio streams for proper sync: [v0, a0, v1, a1, ...]
                concat_inputs = []
                for video_stream, audio_stream in zip(normalized_videos, audio_streams):
                    concat_inputs.extend([video_stream, audio_stream])

                # Concatenate with both video and audio (v=1:a=1) to keep them synchronized
                # Use .node for direct pad access to avoid "multiple outgoing edges" error
                joined = ffmpeg.concat(*concat_inputs, n=len(normalized_videos), v=1, a=1).node
                final_video = joined[0]  # Video output pad
                final_audio = joined[1]  # Audio output pad

                logger.info(f"[Video Editor] Using synchronized concat (v=1:a=1) for {len(normalized_videos)} videos with hard cuts")
                print(f"[Video Editor Tools] Concatenated {len(normalized_videos)} videos with hard cuts (synchronized)")
            else:
                # Video-only concatenation (a=0)
                output = ffmpeg.concat(*normalized_videos, n=len(normalized_videos), v=1, a=0)
                final_video = output
                final_audio = None

                logger.info(f"[Video Editor] Using video-only concat (v=1:a=0) for {len(normalized_videos)} videos")
                print(f"[Video Editor Tools] Concatenated {len(normalized_videos)} videos (video-only)")
        else:
            # Apply xfade transitions between clips
            print(f"[Video Editor Tools] Applying {len(xfade_transitions)} xfade transitions")

            # video_durations already calculated earlier (line ~247)

            # Build complete video/audio chain processing ALL clips
            current_video = normalized_videos[0]
            current_audio = audio_streams[0] if process_audio else None
            offset = video_durations[0]
            logger.info(f"[Video Editor] Starting complete chain - processing ALL {len(normalized_videos)} clips")

            for i in range(len(normalized_videos) - 1):
                # Find if transition exists between clip i and i+1
                trans_info = next((t for t in xfade_transitions if t["index"] == i), None)

                if trans_info:
                    # Apply xfade transition
                    trans_type = trans_info["type"]
                    trans_dur = trans_info["duration"]

                    logger.info(f"[Video Editor]   Clip {i}→{i+1}: xfade {trans_type} at offset {offset}s")

                    # Calculate the actual transition offset (before adjustment)
                    transition_offset = offset
                    offset -= trans_dur

                    current_video = ffmpeg.filter(
                        [current_video, normalized_videos[i + 1]],
                        'xfade',
                        transition=trans_type,
                        duration=trans_dur,
                        offset=offset
                    )

                    # Audio crossfade only if processing audio
                    if process_audio:
                        current_audio = ffmpeg.filter(
                            [current_audio, audio_streams[i + 1]],
                            'acrossfade',
                            d=trans_dur
                        )
                    offset += video_durations[i + 1]
                    logger.info(f"[Video Editor]     → Offset now {offset}s after adding video {i+1} duration")
                else:
                    # Hard cut via concat
                    # Note: Previously used .split()[0] and .asplit()[0] to avoid FFmpeg "multiple outgoing edges"
                    # errors, but this didn't solve the root issue (chained concat topology in mixed scenarios).
                    # Removed split calls - may help with the error, won't break pure hard cut or pure xfade cases.
                    logger.info(f"[Video Editor]   Clip {i}→{i+1}: hard cut (concat) at offset {offset}s")
                    current_video = ffmpeg.concat(current_video, normalized_videos[i + 1], v=1, a=0)
                    if process_audio:
                        current_audio = ffmpeg.concat(current_audio, audio_streams[i + 1], v=0, a=1)
                    offset += video_durations[i + 1]
                    logger.info(f"[Video Editor]     → Offset now {offset}s after adding video {i+1} duration")

            final_video = current_video
            final_audio = current_audio
            logger.info(f"[Video Editor] ✓ Combined ALL {len(normalized_videos)} videos ({len(xfade_transitions)} xfades, {len(normalized_videos)-1-len(xfade_transitions)} hard cuts)")

            # Apply fade_out to final video if it was skipped earlier due to incoming xfade
            if fade_out_duration > 0 and any(t["index"] == len(video_filenames) - 2 for t in xfade_transitions):
                # Calculate total duration and apply timed fade at the end
                total_duration = offset  # This is the final offset after all clips
                fade_start_time = total_duration - fade_out_duration
                logger.info(f"[Video Editor] Applying timed fade_out to final video: starts at {fade_start_time}s, duration {fade_out_duration}s")
                final_video = final_video.filter('fade', type='out', start_time=fade_start_time, duration=fade_out_duration)

        # Apply pixel format conversion after concatenation to avoid graph topology issues
        final_video = final_video.filter('format', 'yuv420p')

        # Output mode: pipe to memory or write to file
        if use_buffer:
            # Buffer mode: Balance quality and speed for preview/streaming
            if process_audio:
                output = ffmpeg.output(
                    final_video,
                    final_audio,
                    'pipe:',
                    format='mp4',
                    vcodec='libx264',
                    acodec='aac',
                    preset='ultrafast',  # Speed priority for real-time streaming
                    crf=23,  # Reasonable quality for preview
                    movflags='frag_keyframe+empty_moov',
                    shortest=None,  # Explicitly use longest stream (video)
                    **{'b:a': '192k'}  # Decent audio bitrate for preview
                )
            else:
                # Video-only output
                output = ffmpeg.output(
                    final_video,
                    'pipe:',
                    format='mp4',
                    vcodec='libx264',
                    preset='ultrafast',
                    crf=23,
                    movflags='frag_keyframe+empty_moov'
                )
        else:
            # File mode: Maximum quality for final output
            if process_audio:
                output = ffmpeg.output(
                    final_video,
                    final_audio,
                    output_path,
                    vcodec='libx264',
                    acodec='aac',
                    preset='veryfast',  # Lower CPU usage with minimal quality impact
                    crf=18,  # Near-lossless quality (visually perfect)
                    movflags='faststart',  # Enable web streaming optimization
                    shortest=None,  # Explicitly use longest stream (video)
                    **{'b:a': '320k'}  # High audio bitrate
                )
            else:
                # Video-only output
                output = ffmpeg.output(
                    final_video,
                    output_path,
                    vcodec='libx264',
                    preset='veryfast',
                    crf=18,
                    movflags='faststart'
                )

        # Run ffmpeg command
        try:
            if use_buffer:
                # Run with pipe to capture stdout (zero-latency mode)
                logger.info(f"[Video Editor] Running FFmpeg (streaming to memory)...")
                process = ffmpeg.run_async(output, pipe_stdout=True, pipe_stderr=True)
                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown FFmpeg error"
                    logger.error(f"[Video Editor] FFmpeg error with code {process.returncode}")
                    logger.error(f"[Video Editor] Error (first 500 chars): {error_msg[:500]}")
                    logger.error(f"[Video Editor] Error (last 500 chars): {error_msg[-500:]}")
                    return {
                        "status": "error",
                        "message": f"FFmpeg error: {error_msg[-800:]}"  # Use tail where actual error is
                    }

                # Write to buffer
                output_buffer.write(stdout)
                output_buffer.seek(0)
                logger.info(f"[Video Editor] ✓ Video combined successfully")
                logger.info("=" * 80)

                return {
                    "status": "success",
                    "output_buffer": output_buffer,
                    "videos_combined": len(video_specs),
                    "transitions_applied": len(xfade_transitions) + (1 if fade_in_duration else 0) + (1 if fade_out_duration else 0)
                }
            else:
                # Traditional file mode - capture stderr for debugging
                logger.info(f"[Video Editor] Running FFmpeg (writing to file)...")

                # Log generated FFmpeg command for debugging
                try:
                    cmd_args = output.get_args()
                    logger.info(f"[Video Editor] FFmpeg command: ffmpeg {' '.join(cmd_args)}")
                except Exception as e:
                    logger.warning(f"[Video Editor] Could not log FFmpeg command: {e}")

                process = ffmpeg.run_async(output, pipe_stderr=True, overwrite_output=True)
                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown FFmpeg error"
                    logger.error(f"[Video Editor] FFmpeg error with code {process.returncode}")
                    logger.error(f"[Video Editor] Error output:\n{error_msg}")
                    return {
                        "status": "error",
                        "message": f"FFmpeg error: {error_msg[-800:]}"
                    }

                # Log stderr even on success to see warnings/info
                if stderr:
                    stderr_msg = stderr.decode('utf-8', errors='ignore')
                    logger.info(f"[Video Editor] FFmpeg output (last 1000 chars):\n{stderr_msg[-1000:]}")

                return {
                    "status": "success",
                    "output_path": output_path,
                    "videos_combined": len(video_specs),
                    "transitions_applied": len(xfade_transitions) + (1 if fade_in_duration else 0) + (1 if fade_out_duration else 0)
                }

        except ffmpeg.Error as e:
            logger.error(f"[Video Editor] FFmpeg error occurred")
            error_output = e.stderr.decode() if e.stderr else str(e)
            # Get last 2000 chars which usually contains the actual error
            error_tail = error_output[-2000:] if len(error_output) > 2000 else error_output
            logger.error(f"[Video Editor] Error details: {error_tail[:1000]}")
            return {
                "status": "error",
                "message": f"FFmpeg error: {error_tail}"
            }
        
    except Exception as e:
        logger.error(f"[Video Editor] Error: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to combine videos: {str(e)}"
        }


def add_audio_to_video(
    video_path: str,
    audio_path: str,
    output_path: Optional[str] = None,
    audio_volume: float = 0.7,
    mix_original: bool = True,
    session_id: str = ""
) -> Dict:
    """
    Add audio/music to a video
    
    Args:
        video_path: Path to input video
        audio_path: Path to audio file
        output_path: Optional output path
        audio_volume: Volume level for added audio (0.0 to 1.0)
        mix_original: Whether to keep original audio
        session_id: Session ID for GCS operations
        
    Returns:
        Dict with status and output path
    """
    
    if not video_path.startswith(("gs://", "https://")) and not os.path.exists(video_path):
        return {"status": "error", "message": f"Video not found: {video_path}"}
    if not audio_path.startswith(("gs://", "https://")) and not os.path.exists(audio_path):
        return {"status": "error", "message": f"Audio not found: {audio_path}"}
    
    # Use temp file for output
    import tempfile
    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="edited_audio_")
        os.close(fd)
    
    try:
        # Load video and audio (convert GCS URLs to signed URLs for FFmpeg)
        prepared_video_path = prepare_media_for_ffmpeg(video_path)
        prepared_audio_path = prepare_media_for_ffmpeg(audio_path)

        # Probe video duration to ensure audio doesn't extend it with black frames
        video_probe = ffmpeg.probe(prepared_video_path)
        video_duration = float(video_probe['format']['duration'])
        logger.info(f"[Video Editor] Video duration: {video_duration:.2f}s")

        # Load video
        video = ffmpeg.input(prepared_video_path)

        # Load audio with duration constraint (trim to video length to prevent black frame padding)
        audio = ffmpeg.input(prepared_audio_path, ss=0, t=video_duration)
        logger.info(f"[Video Editor] Audio trimmed to match video duration: {video_duration:.2f}s")
        
        # Adjust audio volume
        audio_adjusted = audio.audio.filter('volume', audio_volume)

        if mix_original:
            # Check if video has audio stream (probe already done above)
            has_audio = any(s.get('codec_type') == 'audio' for s in video_probe.get('streams', []))

            if has_audio:
                # Mix original audio with new audio using amix
                # duration='shortest' stops mixing when either audio ends (video or music)
                # normalize=0 prevents automatic volume adjustment that can cause distortion
                mixed_audio = ffmpeg.filter(
                    [video.audio, audio_adjusted],
                    'amix',
                    inputs=2,
                    duration='shortest',
                    dropout_transition=0,
                    normalize=0
                )
            else:
                # No original audio, just use adjusted music
                logger.info("[Video Editor] Video has no audio, using only music track")
                mixed_audio = audio_adjusted

            # Output: Audio already trimmed to video duration (prevents black frame padding)
            output = ffmpeg.output(
                video.video,
                mixed_audio,
                output_path,
                vcodec='libx264',  # Re-encode required for filter outputs (volume, amix)
                preset='veryfast',  # Lower CPU usage with minimal quality impact
                crf=18,  # High quality (visually lossless)
                acodec='aac',
                movflags='faststart',  # Enable web streaming
                **{'b:a': '320k'}  # High quality audio (correct ffmpeg-python syntax)
            )
        else:
            # Replace audio entirely (audio already trimmed to video duration)
            output = ffmpeg.output(
                video.video,
                audio_adjusted,
                output_path,
                vcodec='libx264',  # Re-encode required for filter outputs (volume)
                preset='veryfast',  # Lower CPU usage with minimal quality impact
                crf=18,  # High quality (visually lossless)
                acodec='aac',
                movflags='faststart',  # Enable web streaming
                **{'b:a': '320k'}  # High quality audio (correct ffmpeg-python syntax)
            )
        
        # Run ffmpeg command with verbose logging
        logger.info(f"[Video Editor] Running FFmpeg for audio mixing...")
        logger.info(f"[Video Editor] Audio mix config: video={video_path}, audio={audio_path}, volume={audio_volume}, mix_original={mix_original}")

        # Log generated FFmpeg command for debugging
        try:
            cmd_args = output.get_args()
            logger.info(f"[Video Editor] FFmpeg command: ffmpeg {' '.join(cmd_args)}")
        except Exception as e:
            logger.warning(f"[Video Editor] Could not log FFmpeg command: {e}")

        try:
            process = ffmpeg.run_async(output, pipe_stderr=True, overwrite_output=True)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown FFmpeg error"
                logger.error(f"[Video Editor] Audio mixing FFmpeg error with code {process.returncode}")
                logger.error(f"[Video Editor] Audio mixing error output:\n{error_msg}")
                return {
                    "status": "error",
                    "message": f"FFmpeg audio mixing error: {error_msg[-800:]}"
                }

            # Log stderr even on success to see warnings/info
            if stderr:
                stderr_msg = stderr.decode('utf-8', errors='ignore')
                logger.info(f"[Video Editor] Audio mixing FFmpeg output (last 1000 chars):\n{stderr_msg[-1000:]}")

            return {
                "status": "success",
                "output_path": output_path,
                "audio_added": audio_path,
                "volume": audio_volume,
                "mixed_with_original": mix_original
            }
        except ffmpeg.Error as e:
            error_output = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"[Video Editor] Audio mixing FFmpeg error:\n{error_output}")
            return {
                "status": "error",
                "message": f"FFmpeg audio mixing error: {error_output[-800:]}"
            }

    except Exception as e:
        logger.error(f"[Video Editor] Audio mixing exception: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Failed to add audio: {str(e)}"
        }
