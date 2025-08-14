"""
Image utility functions for video generation workflow

Handles image format conversions, URL generation, and uploads for API compatibility
"""

import os
import base64
import logging
from pathlib import Path
from datetime import timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def image_to_base64(image_ref: str) -> str:
    """
    Convert image to Base64 encoded string
    Handles both GCS URLs and local file paths

    Args:
        image_ref: GCS URL or local file path

    Returns:
        Base64 encoded image string (without data: prefix)
    """
    try:
        # GCS URL - download directly to memory (no temp file)
        if image_ref.startswith(("gs://", "https://storage.googleapis.com")):
            from ...storage.gcs_utils import download_to_bytes

            image_bytes = download_to_bytes(image_ref)
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            logger.info(f"GCS image converted to Base64")
            return base64_string

        # Local file path
        else:
            with open(image_ref, "rb") as image_file:
                base64_string = base64.b64encode(image_file.read()).decode('utf-8')
                logger.info(f"Image converted to Base64")
                return base64_string

    except Exception as e:
        logger.error(f"Error converting image to Base64: {str(e)}")
        raise


def get_image_url_for_api(image_ref: str, expiration_hours: int = 168) -> str:
    """
    Convert image reference (GCS or local) to URL for external APIs
    Optimized for speed - generates signed URLs for GCS assets without download/re-upload

    Args:
        image_ref: Image reference (GCS URL, local path)
        expiration_hours: Hours until URL expiration (default 168 = 7 days, max 168)

    Returns:
        Signed URL for the image
    """
    try:
        # GCS URL - generate signed URL directly (fast, no download/upload!)
        if image_ref.startswith(("gs://", "https://storage.googleapis.com")):
            from ...storage.gcs_utils import generate_signed_url

            # Convert https:// format to gs:// if needed
            gs_path = image_ref
            if image_ref.startswith("https://storage.googleapis.com"):
                parts = image_ref.replace("https://storage.googleapis.com/", "").split("?")[0].split("/", 1)
                gs_path = f"gs://{parts[0]}/{parts[1]}"

            # Generate signed URL (metadata operation only, ~50-100ms)
            expiration_days = min(expiration_hours // 24, 7)
            signed_url = generate_signed_url(gs_path, expiration_days=expiration_days)

            if signed_url:
                logger.info(f"Generated signed URL for GCS asset (no upload needed)")
                return signed_url
            else:
                raise ValueError(f"Error generating signed URL for: {gs_path}")

        # Local file - upload to GCS first
        elif os.path.exists(image_ref):
            logger.info(f"Uploading local file to GCS: {image_ref}")
            return upload_image_to_url(image_ref)

        else:
            raise ValueError(f"Invalid image reference: {image_ref}")

    except Exception as e:
        logger.error(f"Error getting image URL: {str(e)}")
        raise


def upload_image_to_url(image_path: str) -> str:
    """
    Upload local image to cloud storage and return public URL
    Uses Google Cloud Storage from project configuration

    Args:
        image_path: Local path to image file

    Returns:
        Public URL for the uploaded image
    """
    try:
        from google.cloud import storage
        import uuid
        from ...core.config import CREDENTIALS, SESSION_BUCKET_NAME

        # Initialize storage client
        storage_client = storage.Client(credentials=CREDENTIALS)
        bucket = storage_client.bucket(SESSION_BUCKET_NAME)

        # Generate unique blob name
        file_extension = Path(image_path).suffix
        blob_name = f"generated/frames/{uuid.uuid4()}{file_extension}"

        # Upload file
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(image_path)

        # Return gs:// path instead of signed URL
        # External APIs that need signed URLs will call generate_signed_url
        gs_path = f"gs://{SESSION_BUCKET_NAME}/{blob_name}"

        logger.info(f"Image uploaded to GCS: {blob_name}")
        logger.info(f"Returning gs:// path (50 chars) instead of signed URL (500+ chars): {gs_path}")
        return gs_path

    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        raise


def validate_image_url(url: str, timeout: int = 10) -> bool:
    """
    Validate that an image URL is accessible via HEAD request

    Args:
        url: Image URL to validate
        timeout: Request timeout in seconds

    Returns:
        True if URL is accessible, False otherwise
    """
    import requests

    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Image URL validation error: {str(e)}")
        return False


def correct_aspect_ratio(image_bytes: bytes, target_aspect_ratio: str) -> bytes:
    """
    Correct image aspect ratio to exact 16:9 or 9:16 using center crop.

    Processes in-memory bytes (no file I/O). Returns original bytes if already correct.
    Used for storyboard frames to ensure video generation compatibility.

    Args:
        image_bytes: Raw image bytes (PNG/JPEG)
        target_aspect_ratio: "horizontal" (16:9) or "vertical" (9:16)

    Returns:
        Corrected image bytes
    """
    from PIL import Image
    from io import BytesIO

    # Define target ratios
    # Currently only supports 16:9 and 9:16 (video generation requirements)
    # Revise this mapping if additional aspect ratios are needed in the future
    target_ratios = {
        "horizontal": 16 / 9,  # 1.7778
        "vertical": 9 / 16     # 0.5625
    }

    target_ratio = target_ratios.get(target_aspect_ratio)
    if not target_ratio:
        logger.warning(f"Unknown aspect ratio '{target_aspect_ratio}', returning original")
        return image_bytes

    try:
        # Load image from bytes
        img = Image.open(BytesIO(image_bytes))
        current_ratio = img.width / img.height

        # Check if already correct (within 0.1% tolerance)
        tolerance = 0.001
        if abs(current_ratio - target_ratio) < tolerance:
            logger.info(f"Aspect ratio already correct ({img.width}x{img.height}), skipping correction")
            return image_bytes

        # Calculate crop box (center crop)
        if current_ratio > target_ratio:
            # Image too wide, crop horizontally
            new_width = int(img.height * target_ratio)
            left = (img.width - new_width) // 2
            crop_box = (left, 0, left + new_width, img.height)
            logger.info(f"Cropping width: {img.width}x{img.height} → {new_width}x{img.height}")
        else:
            # Image too tall, crop vertically
            new_height = int(img.width / target_ratio)
            top = (img.height - new_height) // 2
            crop_box = (0, top, img.width, top + new_height)
            logger.info(f"Cropping height: {img.width}x{img.height} → {img.width}x{new_height}")

        # Apply crop
        cropped_img = img.crop(crop_box)

        # Save to bytes
        output = BytesIO()
        cropped_img.save(output, format='PNG')
        corrected_bytes = output.getvalue()

        logger.info(f"Aspect ratio corrected to {target_aspect_ratio} ({cropped_img.width}x{cropped_img.height})")
        return corrected_bytes

    except Exception as e:
        logger.error(f"Error correcting aspect ratio: {str(e)}, returning original")
        return image_bytes
