"""ElevenLabs API client for audio generation"""

import os
import requests
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElevenLabsClient:
    """Client for ElevenLabs music generation API"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize ElevenLabs client

        Args:
            api_key: ElevenLabs API key (overrides config)
        """
        from ...core.config import ELEVENLABS_API_KEY, ELEVENLABS_API_ENDPOINT

        self.api_key = api_key or ELEVENLABS_API_KEY
        if not self.api_key:
            raise ValueError("ElevenLabs API key not configured. Set ELEVENLABS_API_KEY in environment")

        self.base_url = ELEVENLABS_API_ENDPOINT
        logger.info(f"ElevenLabsClient initialized with endpoint: {self.base_url}")

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        return {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }

    def generate_music(
        self,
        prompt: str,
        duration_ms: int = 60000,
        force_instrumental: bool = True,
        output_format: str = "mp3_44100_128"
    ) -> bytes:
        """Generate music from text prompt

        Args:
            prompt: Natural language description of desired music
            duration_ms: Duration in milliseconds (default: 60000 = 60 seconds)
            force_instrumental: Remove vocals if True (default: True)
            output_format: Audio codec and quality (default: mp3_44100_128)

        Returns:
            Binary audio data

        Raises:
            Exception: If API request fails
        """
        url = f"{self.base_url}/music"

        # Build request payload
        payload = {
            "prompt": prompt,
            "music_length_ms": duration_ms,
            "model_id": "music_v1",
            "force_instrumental": force_instrumental
        }

        # Add output format as query parameter
        params = {"output_format": output_format}

        logger.info(f"[ElevenLabs] Generating music: {prompt[:50]}... ({duration_ms}ms, {output_format})")

        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                params=params,
                timeout=120  # Music generation can take up to 2 minutes
            )

            response.raise_for_status()

            # Response is binary audio data
            audio_data = response.content
            logger.info(f"[ElevenLabs] Music generated successfully ({len(audio_data)} bytes)")
            return audio_data

        except requests.exceptions.HTTPError as e:
            # Handle specific HTTP errors
            if e.response.status_code == 422:
                error_msg = "Invalid parameters (check subscription tier for high-quality formats)"
            elif e.response.status_code == 401:
                error_msg = "Authentication failed (check API key)"
            else:
                error_msg = f"HTTP {e.response.status_code}: {e.response.text}"

            logger.error(f"[ElevenLabs] Music generation error: {error_msg}")
            raise Exception(f"ElevenLabs API error: {error_msg}")

        except requests.exceptions.RequestException as e:
            logger.error(f"[ElevenLabs] Request error: {str(e)}")
            raise Exception(f"Error generating music: {str(e)}")


def get_elevenlabs_client() -> ElevenLabsClient:
    """Get ElevenLabs client instance"""
    return ElevenLabsClient()
