"""Audio generation tools using ElevenLabs API"""

import os
import logging
from pathlib import Path
from typing import Optional
from io import BytesIO
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_music(
    prompt: str,
    name: str,
    duration_ms: int = 60000,
    force_instrumental: bool = True,
    output_format: str = "mp3_44100_128",
    session_id: Optional[str] = None
) -> str:
    """Generate music and upload to GCS

    Args:
        prompt: Natural language description of desired music
        name: Filename without extension (required)
        duration_ms: Duration in milliseconds (default: 60000 = 60 seconds)
        force_instrumental: Remove vocals if True (default: True)
        output_format: Audio codec and quality (default: mp3_44100_128)
        session_id: Session ID for organized storage (optional)

    Returns:
        GCS signed URL (7-day expiration) for the generated music

    Raises:
        Exception: If generation or upload fails
    """
    from .client_elevenlabs import get_elevenlabs_client
    from google.cloud import storage
    from ...core.config import CREDENTIALS, SESSION_BUCKET_NAME, DEPLOYMENT_ENV, MUSIC_GENERATION_CONFIG

    # Input validation
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    max_duration = MUSIC_GENERATION_CONFIG.get("max_duration_ms", 180000)
    if duration_ms <= 0:
        raise ValueError(f"Duration must be positive, got {duration_ms}ms")
    if duration_ms > max_duration:
        raise ValueError(f"Duration {duration_ms}ms exceeds maximum {max_duration}ms")

    try:
        # Generate music
        client = get_elevenlabs_client()
        audio_data = client.generate_music(
            prompt=prompt,
            duration_ms=duration_ms,
            force_instrumental=force_instrumental,
            output_format=output_format
        )

        # Map output format to file extension and content type
        if output_format.startswith("mp3"):
            extension = "mp3"
            content_type = "audio/mpeg"
        elif output_format.startswith("pcm"):
            extension = "wav"
            content_type = "audio/wav"
        elif output_format.startswith("opus"):
            extension = "opus"
            content_type = "audio/opus"
        elif output_format.startswith("ulaw") or output_format.startswith("alaw"):
            extension = "wav"
            content_type = "audio/wav"
        else:
            # Fallback
            extension = "mp3"
            content_type = "audio/mpeg"

        # Upload to GCS
        storage_client = storage.Client(credentials=CREDENTIALS)
        bucket = storage_client.bucket(SESSION_BUCKET_NAME)

        # Create organized path
        environment = DEPLOYMENT_ENV.lower() if DEPLOYMENT_ENV else 'local'
        filename = f"{name}.{extension}"

        if session_id:
            blob_name = f"{environment}/audio/music/{session_id}/{filename}"
        else:
            blob_name = f"{environment}/audio/music/{filename}"

        # Upload audio data
        blob = bucket.blob(blob_name)
        blob.upload_from_file(BytesIO(audio_data), content_type=content_type)

        # Return gs:// path instead of signed URL
        gs_path = f"gs://{SESSION_BUCKET_NAME}/{blob_name}"

        logger.info(f"Music uploaded to GCS: {blob_name}")
        logger.info(f"Music gs:// path (50 chars vs 500+ for signed URL): {gs_path}")
        return gs_path

    except Exception as e:
        logger.error(f"Music generation error: {str(e)}")
        raise Exception(f"Error generating music: {str(e)}")


def generate_music_with_plan(
    composition_plan: dict,
    duration_ms: int = 60000,
    force_instrumental: bool = True,
    output_format: str = "mp3_44100_128",
    session_id: Optional[str] = None
) -> str:
    """Generate music using structured composition plan

    Args:
        composition_plan: Structured musical specification with sections and styles
        duration_ms: Duration in milliseconds
        force_instrumental: Remove vocals if True
        output_format: Audio codec and quality
        session_id: Session ID for organized storage

    Returns:
        GCS signed URL for the generated music

    Note:
        This is a placeholder for future advanced composition features.
        Currently not implemented - use generate_music() with prompts instead.
    """
    raise NotImplementedError(
        "Composition plan generation not yet implemented. Use generate_music() with prompts."
    )
