"""
Google Veo video generation using google-genai SDK
"""

import time
import logging
from typing import Dict, Any, Optional
from google import genai
from google.genai import types
from google.genai.types import GenerateVideosConfig, Image

logger = logging.getLogger(__name__)


class GoogleVeoGenerator:
    """Google Veo video generation via official Vertex AI SDK"""

    def __init__(self, project_id: str = None, location: str = "us-central1"):
        from ...core.config import PROJECT_ID, CREDENTIALS
        self.project_id = project_id or PROJECT_ID
        self.credentials = CREDENTIALS
        self.location = location
        self._client = None

    @property
    def client(self):
        """Lazy client initialization"""
        if self._client is None:
            self._client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
                credentials=self.credentials
            )
        return self._client

    def generate_video(
        self,
        prompt: str,
        image_uri: Optional[str] = None,
        last_frame_uri: Optional[str] = None,
        model: str = "veo-3.1-generate-preview",
        aspect_ratio: str = "16:9",
        duration_seconds: int = 8,
        output_gcs_uri: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Submit video generation task to Google Veo

        Args:
            prompt: Video generation prompt describing the motion
            image_uri: GCS URI for starting frame (gs://bucket/path)
            last_frame_uri: Optional GCS URI for ending frame (dual-frame mode)
            model: Veo model name (default: veo-3.0-generate-preview)
            aspect_ratio: Video aspect ratio (16:9 or 9:16)
            duration_seconds: Video duration in seconds
            output_gcs_uri: GCS URI prefix for output video

        Returns:
            Dict with code (0=success), data (task_id), and message
        """
        try:
            logger.info(f"[GoogleVeo] Submitting video generation task")
            logger.info(f"[GoogleVeo] Model: {model}, Aspect ratio: {aspect_ratio}, Duration: {duration_seconds}s")
            logger.info(f"[GoogleVeo] Prompt: {prompt[:200]}...")

            # Build configuration - last_frame goes INSIDE config per official docs
            config_params = {
                "aspect_ratio": aspect_ratio,
                "output_gcs_uri": output_gcs_uri,
                "duration_seconds": duration_seconds
            }

            # Add last_frame to config if provided (dual-frame mode)
            if last_frame_uri:
                mime_type = "image/png" if last_frame_uri.endswith(".png") else "image/jpeg"
                config_params["last_frame"] = Image(gcs_uri=last_frame_uri, mime_type=mime_type)
                logger.info(f"[GoogleVeo] End frame (dual-frame mode): {last_frame_uri}")

            config = GenerateVideosConfig(**config_params)

            # Build request parameters
            request_params = {
                "model": model,
                "prompt": prompt,
                "config": config
            }

            # Add start frame if provided
            if image_uri:
                mime_type = "image/png" if image_uri.endswith(".png") else "image/jpeg"
                request_params["image"] = Image(gcs_uri=image_uri, mime_type=mime_type)
                logger.info(f"[GoogleVeo] Start frame: {image_uri}")

            # Submit generation request (returns long-running operation)
            operation = self.client.models.generate_videos(**request_params)

            logger.info(f"[GoogleVeo] Task submitted successfully: {operation.name}")

            return {
                "code": 0,
                "data": {
                    "task_id": operation.name,
                    "task_status": "submitted"
                },
                "message": "Success"
            }

        except Exception as e:
            logger.error(f"[GoogleVeo] Error submitting task: {str(e)}")
            return {
                "code": -1,
                "data": {"task_id": None, "task_status": "failed"},
                "message": f"Google Veo error: {str(e)}"
            }

    def query_task(self, operation_name: str) -> Dict[str, Any]:
        """
        Query task status using operation name

        Args:
            operation_name: The operation name returned from generate_video

        Returns:
            Dict with code, data (task_id, task_status, task_result), and message
        """
        try:
            # Reconstruct operation object from name (per official docs pattern)
            operation = types.GenerateVideosOperation(name=operation_name)
            # Refresh operation status from API
            operation = self.client.operations.get(operation)

            if operation.done:
                if operation.response:
                    # Success - extract video URL using .result (Vertex AI pattern)
                    # Note: .response is checked as boolean, .result contains actual data
                    videos = operation.result.generated_videos
                    if videos:
                        video_url = videos[0].video.uri
                        logger.info(f"[GoogleVeo] Task completed: {video_url}")
                        return {
                            "code": 0,
                            "data": {
                                "task_id": operation_name,
                                "task_status": "succeed",
                                "task_result": {
                                    "videos": [{"url": video_url}]
                                }
                            },
                            "message": "Success"
                        }
                    else:
                        logger.warning(f"[GoogleVeo] Task completed but no videos in result")
                        return {
                            "code": -1,
                            "data": {
                                "task_id": operation_name,
                                "task_status": "failed"
                            },
                            "message": "No videos in result"
                        }
                else:
                    # Operation done but with error
                    error_msg = str(operation.error) if operation.error else "Unknown error"
                    logger.error(f"[GoogleVeo] Task failed: {error_msg}")
                    return {
                        "code": -1,
                        "data": {
                            "task_id": operation_name,
                            "task_status": "failed"
                        },
                        "message": error_msg
                    }
            else:
                # Still processing
                logger.info(f"[GoogleVeo] Task still processing: {operation_name}")
                return {
                    "code": 0,
                    "data": {
                        "task_id": operation_name,
                        "task_status": "processing"
                    },
                    "message": "Processing"
                }

        except Exception as e:
            logger.error(f"[GoogleVeo] Error querying task {operation_name}: {str(e)}")
            return {
                "code": -1,
                "data": {
                    "task_id": operation_name,
                    "task_status": "failed"
                },
                "message": f"Query error: {str(e)}"
            }

    def wait_for_completion(
        self,
        operation_name: str,
        max_wait_time: int = 600,
        poll_interval: int = 15
    ) -> Dict[str, Any]:
        """
        Wait for video generation to complete

        Args:
            operation_name: The operation name to poll
            max_wait_time: Maximum wait time in seconds (default: 600)
            poll_interval: Time between polls in seconds (default: 15)

        Returns:
            Final task result dict
        """
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            result = self.query_task(operation_name)

            status = result.get("data", {}).get("task_status")

            if status == "succeed":
                return result
            elif status == "failed":
                return result

            # Still processing
            elapsed = int(time.time() - start_time)
            logger.info(f"[GoogleVeo] Waiting... ({elapsed}s elapsed)")
            time.sleep(poll_interval)

        # Timeout
        logger.error(f"[GoogleVeo] Timeout after {max_wait_time}s waiting for {operation_name}")
        return {
            "code": -1,
            "data": {
                "task_id": operation_name,
                "task_status": "timeout"
            },
            "message": f"Timeout after {max_wait_time}s"
        }
