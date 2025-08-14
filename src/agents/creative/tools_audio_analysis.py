"""
Audio beat analysis tools using Librosa
Analyzes music tracks for beats, energy, and tempo
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import librosa
import numpy as np
import requests
from scipy.signal import find_peaks

from ...core.state import RAGState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_music_beats(
    audio_url: str,
    state: Optional[RAGState] = None,
    energy_interval: float = 0.5
) -> Dict[str, Any]:
    """
    Analyze music track for beat timing, energy, and tempo using Librosa.

    Provides raw informational data for LLM agent to reason about.
    No helper functions - agent makes all editing decisions.

    Args:
        audio_url: GCS signed URL or local file path to audio
        state: RAGState for caching analysis results
        energy_interval: Time interval (seconds) between energy samples (default: 0.5s)
                        0.5s = 2 samples/second (good for video editing)

    Returns:
        Complete audio analysis:
        {
            "track_name": "song_name",
            "url": "https://...",
            "duration_seconds": 60.0,
            "tempo_bpm": 128.5,
            "beat_timestamps": [0.467, 0.934, ...],
            "downbeat_timestamps": [0.467, 2.335, ...],
            "energy_timeline": {
                "timestamps": [0.0, 0.5, ...],
                "values": [0.12, 0.25, ...]
            },
            "energy_peaks": {
                "timestamps": [8.2, 16.5, ...],
                "intensity": [0.92, 0.88, ...]
            },
            "analysis_timestamp": "ISO timestamp"
        }
    """
    try:
        logger.info("=" * 80)
        logger.info(f"[Audio Analysis] Starting analysis for: {audio_url}")
        logger.info("=" * 80)

        # Extract track name from URL or path
        track_name = _extract_track_name(audio_url)

        # Check cache
        if state and "audio_beat_analysis" in state:
            cached = state.get("audio_beat_analysis", {}).get(track_name)
            if cached:
                logger.info(f"[Audio Analysis] ✓ Using cached result for '{track_name}'")
                return cached

        # Download or load audio file
        audio_path = _get_audio_file(audio_url)

        try:
            # Load audio with librosa (preserve original sample rate)
            logger.info(f"[Audio Analysis] Loading audio file...")
            y, sr = librosa.load(audio_path, sr=None)
            duration = float(librosa.get_duration(y=y, sr=sr))
            logger.info(f"[Audio Analysis] ✓ Loaded: {duration:.1f}s @ {sr} Hz")

            # Beat detection
            logger.info(f"[Audio Analysis] Detecting beats and tempo...")
            tempo, beat_frames = librosa.beat.beat_track(
                y=y,
                sr=sr,
                start_bpm=120.0,
                tightness=100
            )

            # Convert tempo to scalar if it's an array
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo.item()) if tempo.size == 1 else float(np.mean(tempo))
            else:
                tempo = float(tempo)

            # Round beat timestamps to 2 decimals (10ms precision)
            beat_timestamps = [round(t, 2) for t in librosa.frames_to_time(beat_frames, sr=sr).tolist()]
            logger.info(f"[Audio Analysis] ✓ Tempo: {tempo:.1f} BPM, Beats: {len(beat_timestamps)}")

            # Downbeat detection (every 4th beat in 4/4 time)
            downbeat_timestamps = beat_timestamps[::4]
            logger.info(f"[Audio Analysis] ✓ Downbeats: {len(downbeat_timestamps)} (every 4th beat)")

            # Energy analysis using RMS
            logger.info(f"[Audio Analysis] Computing energy levels...")
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            rms_times_full = librosa.frames_to_time(
                np.arange(len(rms)),
                sr=sr,
                hop_length=512
            )

            # Normalize RMS to 0-1 scale
            rms_max = np.max(rms)
            if rms_max > 0:
                rms_normalized = rms / rms_max
            else:
                rms_normalized = rms

            # Downsample energy timeline to practical intervals
            # Create time grid at specified intervals
            num_samples = int(duration / energy_interval) + 1
            energy_times = np.linspace(0, duration, num_samples)

            # Interpolate RMS values at these times
            energy_values = np.interp(energy_times, rms_times_full, rms_normalized)

            # Round for readability
            energy_times_rounded = [round(t, 2) for t in energy_times.tolist()]
            energy_values_rounded = [round(v, 3) for v in energy_values.tolist()]

            logger.info(f"[Audio Analysis] ✓ Energy timeline: {len(energy_times_rounded)} points @ {energy_interval}s intervals")

            # Energy peak detection
            logger.info(f"[Audio Analysis] Detecting energy peaks...")
            peak_threshold = np.mean(rms) + np.std(rms)
            peak_distance = sr // 2  # Minimum 0.5s between peaks
            peaks, properties = find_peaks(
                rms,
                height=peak_threshold,
                distance=peak_distance
            )

            # Round peak data
            peak_timestamps = [round(t, 2) for t in librosa.frames_to_time(peaks, sr=sr, hop_length=512).tolist()]
            peak_intensities = [round(i, 3) for i in (properties['peak_heights'] / rms_max).tolist()]
            logger.info(f"[Audio Analysis] ✓ Energy peaks: {len(peak_timestamps)} detected")

            # Build analysis result with rounded values and field definitions
            analysis = {
                "track_name": track_name,
                "url": audio_url,
                "duration_seconds": round(duration, 2),
                "tempo_bpm": round(tempo, 1),
                "beat_timestamps": beat_timestamps,
                "downbeat_timestamps": downbeat_timestamps,
                "energy_timeline": {
                    "timestamps": energy_times_rounded,
                    "values": energy_values_rounded,
                    "interval_seconds": energy_interval
                },
                "energy_peaks": {
                    "timestamps": peak_timestamps,
                    "intensity": peak_intensities
                },
                "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
                "_field_definitions": {
                    "track_name": "Audio track filename without extension",
                    "url": "Source audio file path or URL",
                    "duration_seconds": "Total track duration in seconds (2 decimals)",
                    "tempo_bpm": "Detected tempo in beats per minute (1 decimal)",
                    "beat_timestamps": "Timestamps (seconds) of every musical beat (2 decimals). Use for transition timing.",
                    "downbeat_timestamps": "Timestamps (seconds) of strong beats/bar starts (2 decimals). Use for scene changes and impactful transitions.",
                    "energy_timeline": {
                        "description": "Audio intensity over time for pacing decisions",
                        "timestamps": "Time points in seconds (2 decimals)",
                        "values": "Normalized energy levels 0-1 (3 decimals). Higher = louder/more intense.",
                        "interval_seconds": "Time between energy samples (controls granularity)"
                    },
                    "energy_peaks": {
                        "description": "Moments of sudden loudness/intensity for visual peak alignment",
                        "timestamps": "Time points of peaks in seconds (2 decimals)",
                        "intensity": "Peak intensity 0-1 (3 decimals). 1.0 = loudest moment in track."
                    },
                    "analysis_timestamp": "When this analysis was generated (ISO 8601 UTC)"
                }
            }

            # Cache in state
            if state:
                if "audio_beat_analysis" not in state:
                    state["audio_beat_analysis"] = {}
                state["audio_beat_analysis"][track_name] = analysis
                logger.info(f"[Audio Analysis] ✓ Cached result for '{track_name}'")

            logger.info("=" * 80)
            logger.info(f"[Audio Analysis] ✓ Analysis complete for '{track_name}'!")
            logger.info(f"[Audio Analysis]   Duration: {analysis['duration_seconds']:.1f}s")
            logger.info(f"[Audio Analysis]   Tempo: {analysis['tempo_bpm']:.1f} BPM")
            logger.info(f"[Audio Analysis]   Beats: {len(analysis['beat_timestamps'])}")
            logger.info(f"[Audio Analysis]   Downbeats: {len(analysis['downbeat_timestamps'])}")
            logger.info(f"[Audio Analysis]   Energy points: {len(analysis['energy_timeline']['timestamps'])}")
            logger.info(f"[Audio Analysis]   Peaks: {len(analysis['energy_peaks']['timestamps'])}")
            logger.info("=" * 80)

            return analysis

        finally:
            # Cleanup temp file if downloaded
            if audio_path.startswith(tempfile.gettempdir()):
                try:
                    os.remove(audio_path)
                    logger.debug(f"Cleaned up temp file: {audio_path}")
                except Exception as e:
                    logger.warning(f"Error cleaning up temp file: {e}")

    except Exception as e:
        logger.error(f"Audio analysis error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "track_name": _extract_track_name(audio_url),
            "url": audio_url
        }


def _extract_track_name(url_or_path: str) -> str:
    """Extract track name from URL or file path"""
    # Get filename from URL or path
    if url_or_path.startswith("http"):
        path = urlparse(url_or_path).path
    else:
        path = url_or_path

    filename = os.path.basename(path)
    # Remove extension
    name = os.path.splitext(filename)[0]
    return name


def _get_audio_file(audio_url: str) -> str:
    """
    Download audio from URL or return local path.

    Returns:
        Local file path to audio file
    """
    # Local file
    if os.path.exists(audio_url):
        logger.info(f"Using local file: {audio_url}")
        return audio_url

    # Download from URL
    if audio_url.startswith("http"):
        logger.info(f"Downloading audio from URL...")

        # Create temp file with proper extension
        ext = os.path.splitext(urlparse(audio_url).path)[1] or ".mp3"
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=ext
        )
        temp_path = temp_file.name
        temp_file.close()

        # Download
        response = requests.get(audio_url, timeout=60)
        response.raise_for_status()

        with open(temp_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"Downloaded to: {temp_path}")
        return temp_path

    # GCS path (gs://bucket/path)
    if audio_url.startswith("gs://"):
        raise ValueError(
            "GCS paths (gs://) not supported. "
            "Please use signed URLs (https://) instead."
        )

    raise ValueError(f"Unsupported audio URL format: {audio_url}")
