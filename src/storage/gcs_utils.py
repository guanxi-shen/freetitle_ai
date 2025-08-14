"""
Simple GCS utilities for asset storage - builds on existing patterns
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union
from datetime import timedelta
from contextlib import contextmanager
from google.cloud import storage
from ..core.config import CREDENTIALS, SESSION_BUCKET_NAME, PUBLIC_BUCKET_NAME, DEPLOYMENT_ENV


# Initialize storage client once
_storage_client = None
_bucket = None

def _get_bucket():
    """Get or create bucket instance (singleton pattern)"""
    global _storage_client, _bucket
    if _bucket is None:
        _storage_client = storage.Client(credentials=CREDENTIALS)
        _bucket = _storage_client.bucket(SESSION_BUCKET_NAME)
    return _bucket


def upload_bytes_to_gcs(
    data: bytes,
    session_id: str,
    asset_type: str,
    filename: str,
    content_type: Optional[str] = None,
    return_format: str = "gs"
) -> str:
    """
    Upload bytes data to GCS and return path

    Args:
        data: Bytes data to upload
        session_id: Session identifier
        asset_type: Type of asset (characters, storyboards, videos)
        filename: Name of the file
        content_type: Optional MIME type
        return_format: "gs" for gs:// path (default), "signed" for signed URL

    Returns:
        GCS path (gs://) or signed URL based on return_format, empty string on failure
    """
    try:
        print(f"[GCS Utils] DEBUG: upload_bytes_to_gcs called - session_id={session_id}, asset_type={asset_type}, filename={filename}, data_size={len(data)} bytes, return_format={return_format}")
        bucket = _get_bucket()
        print(f"[GCS Utils] DEBUG: Got bucket: {bucket.name if bucket else 'None'}")

        # Build path following existing pattern
        environment = DEPLOYMENT_ENV.lower() if DEPLOYMENT_ENV else 'local'
        blob_name = f"{environment}/sessions/{session_id}/{asset_type}/{filename}"
        print(f"[GCS Utils] DEBUG: Uploading to blob_name: {blob_name}")

        # Upload
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data, content_type=content_type)
        print(f"[GCS Utils] DEBUG: Upload completed successfully")

        # Return format based on parameter
        if return_format == "signed":
            # Generate signed URL (7 days like video_tools.py)
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(days=7),
                method="GET"
            )
            print(f"[GCS Utils] DEBUG: Generated signed URL: {signed_url[:100]}..." if len(signed_url) > 100 else f"[GCS Utils] DEBUG: Generated signed URL: {signed_url}")
            return signed_url
        else:
            # Return gs:// path (default)
            gs_path = f"gs://{SESSION_BUCKET_NAME}/{blob_name}"
            print(f"[GCS Utils] DEBUG: Returning gs:// path: {gs_path}")
            return gs_path

    except Exception as e:
        print(f"[GCS Utils] Error uploading {filename}: {e}")
        import traceback
        print(f"[GCS Utils] DEBUG: Full exception traceback:\n{traceback.format_exc()}")
        return ""


def upload_file_to_gcs(
    file_path: str,
    session_id: str,
    asset_type: str,
    filename: Optional[str] = None,
    return_format: str = "gs"
) -> str:
    """
    Upload file to GCS and return path

    Args:
        file_path: Local file path
        session_id: Session identifier
        asset_type: Type of asset (characters, storyboards, videos)
        filename: Optional custom filename (uses basename if not provided)
        return_format: "gs" for gs:// path (default), "signed" for signed URL

    Returns:
        GCS path (gs://) or signed URL based on return_format, empty string on failure
    """
    try:
        bucket = _get_bucket()

        # Use provided filename or extract from path
        if filename is None:
            filename = Path(file_path).name

        # Build path following existing pattern
        environment = DEPLOYMENT_ENV.lower() if DEPLOYMENT_ENV else 'local'
        blob_name = f"{environment}/sessions/{session_id}/{asset_type}/{filename}"

        # Upload
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)

        # Return format based on parameter
        if return_format == "signed":
            # Generate signed URL (7 days like video_tools.py)
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(days=7),
                method="GET"
            )
            return signed_url
        else:
            # Return gs:// path (default)
            gs_path = f"gs://{SESSION_BUCKET_NAME}/{blob_name}"
            return gs_path

    except Exception as e:
        print(f"[GCS Utils] Error uploading file {file_path}: {e}")
        return ""


def download_to_bytes(gcs_url: str) -> bytes:
    """
    Download GCS file directly to memory (no temp file)

    Args:
        gcs_url: GCS URL (gs:// or signed HTTPS URL)

    Returns:
        File contents as bytes

    Example:
        image_bytes = download_to_bytes(image_url)
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
    """
    if gcs_url.startswith("gs://"):
        # Parse GCS URL
        parts = gcs_url.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""

        bucket = storage.Client(credentials=CREDENTIALS).bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()
    else:
        # Signed URL - use requests
        import requests
        response = requests.get(gcs_url, timeout=60)
        response.raise_for_status()
        return response.content


@contextmanager
def download_to_temp(gcs_url: str, suffix: str = ""):
    """
    Download GCS file to temp file with automatic cleanup
    
    Args:
        gcs_url: GCS URL or signed URL
        suffix: Optional file suffix (e.g., ".mp4")
        
    Yields:
        Path to temporary file
        
    Example:
        with download_to_temp(video_url, ".mp4") as temp_path:
            # Use temp_path for processing
            process_video(temp_path)
        # File automatically deleted after context
    """
    temp_file = None
    try:
        # Create temp file
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)  # Close file descriptor
        temp_file = temp_path
        
        # Download from GCS
        if gcs_url.startswith("gs://"):
            # Parse GCS URL
            parts = gcs_url.replace("gs://", "").split("/", 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ""
            
            bucket = storage.Client(credentials=CREDENTIALS).bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(temp_path)
        else:
            # Signed URL - use requests
            import requests
            response = requests.get(gcs_url, stream=True)
            response.raise_for_status()
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        yield temp_path
        
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass  # Best effort cleanup


def get_gcs_path(session_id: str, asset_type: str, filename: str) -> str:
    """
    Get the GCS path for an asset (without uploading)
    
    Args:
        session_id: Session identifier
        asset_type: Type of asset
        filename: Name of the file
        
    Returns:
        GCS path (gs://...)
    """
    environment = DEPLOYMENT_ENV.lower() if DEPLOYMENT_ENV else 'local'
    return f"gs://{SESSION_BUCKET_NAME}/{environment}/sessions/{session_id}/{asset_type}/{filename}"


def generate_signed_url(gcs_path: str, expiration_days: int = 7) -> str:
    """
    Generate signed URL for existing GCS object

    Args:
        gcs_path: GCS path (gs://...)
        expiration_days: Days until expiration (max 7)

    Returns:
        Signed URL or empty string on failure
    """
    try:
        # Parse GCS path
        if not gcs_path.startswith("gs://"):
            return ""

        parts = gcs_path.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""

        bucket = storage.Client(credentials=CREDENTIALS).bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Generate signed URL
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(days=min(expiration_days, 7)),
            method="GET"
        )

        return signed_url

    except Exception as e:
        print(f"[GCS Utils] Error generating signed URL for {gcs_path}: {e}")
        return ""


def copy_session_assets(session_id: str, save_name: str) -> tuple[int, dict]:
    """
    Copy all session assets from temp to permanent location

    Uses GCS server-side copy (no download/upload bandwidth)

    Args:
        session_id: Source session ID
        save_name: Destination session name

    Returns:
        Tuple of (assets_copied_count, filename_to_path_mapping)
        filename_to_path_mapping maps filename to new gs:// path
    """
    bucket = _get_bucket()
    environment = DEPLOYMENT_ENV.lower() if DEPLOYMENT_ENV else 'local'
    copied_count = 0
    filename_mapping = {}

    # Standard assets under sessions/{id}/
    standard_types = ['characters', 'storyboards', 'references', 'supplementary']
    for asset_type in standard_types:
        source_prefix = f"{environment}/sessions/{session_id}/{asset_type}/"
        dest_prefix = f"{environment}/saved_sessions/{save_name}/{asset_type}/"

        blobs = bucket.list_blobs(prefix=source_prefix)
        for blob in blobs:
            filename = blob.name.split('/')[-1]
            if filename:
                # Copy to destination
                dest_blob_name = f"{dest_prefix}{filename}"
                dest_blob = bucket.copy_blob(blob, bucket, dest_blob_name)

                # Map filename to new gs:// path (not signed URL)
                gs_path = f"gs://{SESSION_BUCKET_NAME}/{dest_blob_name}"
                filename_mapping[filename] = gs_path
                copied_count += 1

    # Videos (special path: videos/{id}/ instead of sessions/{id}/videos/)
    video_source_prefix = f"{environment}/videos/{session_id}/"
    video_dest_prefix = f"{environment}/saved_sessions/{save_name}/videos/"

    blobs = bucket.list_blobs(prefix=video_source_prefix)
    for blob in blobs:
        filename = blob.name.split('/')[-1]
        if filename:
            # Copy to destination
            dest_blob_name = f"{video_dest_prefix}{filename}"
            dest_blob = bucket.copy_blob(blob, bucket, dest_blob_name)

            # Map filename to new gs:// path (not signed URL)
            gs_path = f"gs://{SESSION_BUCKET_NAME}/{dest_blob_name}"
            filename_mapping[filename] = gs_path
            copied_count += 1

    # Audio (special path: audio/music/{id}/ instead of sessions/{id}/audio/)
    audio_source_prefix = f"{environment}/audio/music/{session_id}/"
    audio_dest_prefix = f"{environment}/saved_sessions/{save_name}/audio/"

    blobs = bucket.list_blobs(prefix=audio_source_prefix)
    for blob in blobs:
        filename = blob.name.split('/')[-1]
        if filename:
            # Copy to destination
            dest_blob_name = f"{audio_dest_prefix}{filename}"
            dest_blob = bucket.copy_blob(blob, bucket, dest_blob_name)

            # Map filename to new gs:// path (not signed URL)
            gs_path = f"gs://{SESSION_BUCKET_NAME}/{dest_blob_name}"
            filename_mapping[filename] = gs_path
            copied_count += 1

    print(f"[GCS Utils] Copied {copied_count} assets from session {session_id} to {save_name}")
    print(f"[GCS Utils] Generated {len(filename_mapping)} filename→path mappings")
    return copied_count, filename_mapping


def remap_asset_urls(state: dict, filename_mapping: dict) -> dict:
    """
    Update all asset URLs in state using filename-to-path mapping

    Remaps URLs in state fields:
    1. asset_urls - URLs in list items
    2. character_image_metadata - "path" field
    3. storyboard_frame_metadata - "path" field
    4. supplementary_content_metadata - "path" field
    5. video_generation_tasks - "path" and "video_url" fields
    6. reference_images - direct list of URL strings
    7. edited_videos - "path" and "public_url" fields
    8. image_annotations - "path" field
    9. generated_storyboards - nested "path" fields in storyboards and workspace_frames

    Args:
        state: Session state dict
        filename_mapping: Dict mapping filename to new gs:// path

    Returns:
        Updated state dict with all URLs remapped to new gs:// paths
    """
    def remap_url_string(url_str):
        """Extract filename from URL and lookup new signed URL"""
        if not isinstance(url_str, str):
            return url_str

        # Extract filename from URL: "https://.../path/file.png?sig=..." -> "file.png"
        try:
            filename = url_str.split('/')[-1].split('?')[0]
            if filename and filename in filename_mapping:
                return filename_mapping[filename]
        except:
            pass

        # If extraction fails or not in mapping, return original
        return url_str

    remapped_count = 0

    # 1. Update asset_urls dict
    if "asset_urls" in state and isinstance(state["asset_urls"], dict):
        for asset_type, items in state["asset_urls"].items():
            if isinstance(items, list):
                updated_items = []
                for item in items:
                    if isinstance(item, dict) and "url" in item:
                        updated_item = item.copy()
                        updated_item["url"] = remap_url_string(item["url"])
                        updated_items.append(updated_item)
                        remapped_count += 1
                    else:
                        updated_items.append(remap_url_string(item))
                        remapped_count += 1
                state["asset_urls"][asset_type] = updated_items

    # 2. Update metadata dicts (character, storyboard, supplementary, video, audio)
    for metadata_key in ["character_image_metadata", "storyboard_frame_metadata", "supplementary_content_metadata", "video_generation_metadata", "audio_generation_metadata", "image_annotations"]:
        if metadata_key in state and isinstance(state[metadata_key], dict):
            for key, metadata in state[metadata_key].items():
                if isinstance(metadata, dict):
                    # Update path field
                    if "path" in metadata and metadata["path"]:
                        metadata["path"] = remap_url_string(metadata["path"])
                        remapped_count += 1
                    # Update url field (used by video_generation_metadata)
                    if "url" in metadata and metadata["url"]:
                        metadata["url"] = remap_url_string(metadata["url"])
                        remapped_count += 1

    # 3. Update video_generation_tasks
    if "video_generation_tasks" in state and isinstance(state["video_generation_tasks"], list):
        for task in state["video_generation_tasks"]:
            if isinstance(task, dict):
                if "path" in task and task["path"]:
                    task["path"] = remap_url_string(task["path"])
                    remapped_count += 1
                if "video_url" in task and task["video_url"]:
                    task["video_url"] = remap_url_string(task["video_url"])
                    remapped_count += 1

    # 4. Update reference_images (direct list of URLs)
    if "reference_images" in state and isinstance(state["reference_images"], list):
        state["reference_images"] = [
            remap_url_string(url) for url in state["reference_images"]
        ]
        remapped_count += len(state["reference_images"])

    # 5. Update edited_videos
    if "edited_videos" in state and isinstance(state["edited_videos"], list):
        for video in state["edited_videos"]:
            if isinstance(video, dict):
                if "path" in video and video["path"]:
                    video["path"] = remap_url_string(video["path"])
                    remapped_count += 1
                if "public_url" in video and video["public_url"]:
                    video["public_url"] = remap_url_string(video["public_url"])
                    remapped_count += 1


    # 7. Update generated_storyboards (nested "path" fields)
    if "generated_storyboards" in state and isinstance(state["generated_storyboards"], dict):
        # Update storyboards[i].frames[j].path
        if "storyboards" in state["generated_storyboards"]:
            for storyboard in state["generated_storyboards"]["storyboards"]:
                if isinstance(storyboard, dict) and "frames" in storyboard:
                    for frame in storyboard["frames"]:
                        if isinstance(frame, dict) and "path" in frame and frame["path"]:
                            frame["path"] = remap_url_string(frame["path"])
                            remapped_count += 1

        # Update workspace_frames[i].path
        if "workspace_frames" in state["generated_storyboards"]:
            for frame in state["generated_storyboards"]["workspace_frames"]:
                if isinstance(frame, dict) and "path" in frame and frame["path"]:
                    frame["path"] = remap_url_string(frame["path"])
                    remapped_count += 1

    print(f"[GCS Utils] Remapped {remapped_count} URLs using URL mapping")
    return state


def detect_saved_session_name(state: dict) -> Optional[str]:
    """
    Detect if state contains saved_sessions URLs and extract the save_name

    Checks all state fields that contain URLs:
    1. reference_images
    2. character_image_metadata
    3. storyboard_frame_metadata
    4. supplementary_content_metadata
    5. video_generation_tasks
    6. edited_videos
    7. asset_urls
    8. image_annotations
    9. generated_storyboards

    Args:
        state: Session state dict

    Returns:
        Save name if found, None if state uses temp session URLs
    """
    environment = DEPLOYMENT_ENV.lower() if DEPLOYMENT_ENV else 'local'
    saved_sessions_marker = f"{environment}/saved_sessions/"

    def extract_save_name(url_str):
        """Extract save_name from URL containing saved_sessions marker"""
        if url_str and isinstance(url_str, str) and saved_sessions_marker in url_str:
            try:
                return url_str.split(saved_sessions_marker)[1].split('/')[0]
            except:
                pass
        return None

    # 1. Check reference_images (list of URL strings)
    if "reference_images" in state and isinstance(state["reference_images"], list):
        for url in state["reference_images"]:
            save_name = extract_save_name(url)
            if save_name:
                return save_name

    # 2-4. Check metadata dicts (path field)
    for metadata_key in ["character_image_metadata", "storyboard_frame_metadata", "supplementary_content_metadata", "image_annotations"]:
        if metadata_key in state and isinstance(state[metadata_key], dict):
            for key, metadata in state[metadata_key].items():
                if isinstance(metadata, dict):
                    save_name = extract_save_name(metadata.get("path"))
                    if save_name:
                        return save_name

    # 5. Check video_generation_tasks (list with path/video_url)
    if "video_generation_tasks" in state and isinstance(state["video_generation_tasks"], list):
        for task in state["video_generation_tasks"]:
            if isinstance(task, dict):
                save_name = extract_save_name(task.get("path")) or extract_save_name(task.get("video_url"))
                if save_name:
                    return save_name

    # 6. Check edited_videos (list with path/public_url)
    if "edited_videos" in state and isinstance(state["edited_videos"], list):
        for video in state["edited_videos"]:
            if isinstance(video, dict):
                save_name = extract_save_name(video.get("path")) or extract_save_name(video.get("public_url"))
                if save_name:
                    return save_name

    # 7. Check asset_urls (dict of lists with url field)
    if "asset_urls" in state and isinstance(state["asset_urls"], dict):
        for asset_type, items in state["asset_urls"].items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        save_name = extract_save_name(item.get("url"))
                        if save_name:
                            return save_name


    # 9. Check generated_storyboards (nested path fields)
    if "generated_storyboards" in state and isinstance(state["generated_storyboards"], dict):
        if "storyboards" in state["generated_storyboards"]:
            for storyboard in state["generated_storyboards"]["storyboards"]:
                if isinstance(storyboard, dict) and "frames" in storyboard:
                    for frame in storyboard["frames"]:
                        if isinstance(frame, dict):
                            save_name = extract_save_name(frame.get("path"))
                            if save_name:
                                return save_name

        if "workspace_frames" in state["generated_storyboards"]:
            for frame in state["generated_storyboards"]["workspace_frames"]:
                if isinstance(frame, dict):
                    save_name = extract_save_name(frame.get("path"))
                    if save_name:
                        return save_name

    return None


def regenerate_session_urls(state: dict, save_name: str) -> dict:
    """
    Regenerate all paths in state for a saved session

    Scans saved_sessions/{save_name}/ in GCS for all assets,
    generates fresh gs:// paths, and remaps all URLs in state.

    Args:
        state: Session state dict with potentially expired URLs or old paths
        save_name: Name of saved session

    Returns:
        Updated state with gs:// paths
    """
    bucket = _get_bucket()
    environment = DEPLOYMENT_ENV.lower() if DEPLOYMENT_ENV else 'local'
    filename_mapping = {}

    print(f"[GCS Utils] Regenerating paths for saved session: {save_name}")

    # Asset types to scan
    asset_types = ['characters', 'storyboards', 'references', 'supplementary', 'videos', 'audio']

    for asset_type in asset_types:
        prefix = f"{environment}/saved_sessions/{save_name}/{asset_type}/"
        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            filename = blob.name.split('/')[-1]
            if filename:
                # Map to gs:// path (not signed URL)
                gs_path = f"gs://{SESSION_BUCKET_NAME}/{blob.name}"
                filename_mapping[filename] = gs_path
                print(f"[GCS Utils] Mapped {filename} -> {gs_path}")

    print(f"[GCS Utils] Regenerated {len(filename_mapping)} gs:// paths for session {save_name}")

    # If no assets found, return state unchanged (graceful handling)
    if not filename_mapping:
        print(f"[GCS Utils] No assets found for session {save_name}, state unchanged")
        return state

    # Remap all URLs in state using the filename mapping
    return remap_asset_urls(state, filename_mapping)


def upload_to_public_gcs(gcs_source: str) -> Optional[str]:
    """
    Copy GCS file to public bucket for external APIs
    Uses server-side copy - no data transfer through this machine.

    Args:
        gcs_source: GCS path (gs://bucket/path)

    Returns:
        Public URL or None on failure
    """
    import uuid

    try:
        # Parse source path
        parts = gcs_source.replace("gs://", "").split("/", 1)
        source_bucket_name = parts[0]
        source_blob_name = parts[1]

        # Server-side copy to public bucket
        client = storage.Client(credentials=CREDENTIALS)
        source_bucket = client.bucket(source_bucket_name)
        source_blob = source_bucket.blob(source_blob_name)

        dest_bucket = client.bucket(PUBLIC_BUCKET_NAME)
        dest_blob_name = f"temp/{uuid.uuid4().hex[:12]}.png"

        source_bucket.copy_blob(source_blob, dest_bucket, dest_blob_name)

        return f"https://storage.googleapis.com/{PUBLIC_BUCKET_NAME}/{dest_blob_name}"

    except Exception as e:
        print(f"[GCS Public] Error: {e}")
        return None


def prepare_image_for_llm(gcs_path: str, model_type: str = "gemini") -> Optional[dict]:
    """
    Prepare image for LLM based on model type.

    Gemini: gs:// URI directly (zero latency - Google internal)
    Others: Download + base64

    Args:
        gcs_path: GCS path (gs://bucket/path)
        model_type: "gemini" or other model types

    Returns:
        dict with 'file_uri' (Gemini) or 'data'+'media_type' (others), None on error
    """
    if not gcs_path or not gcs_path.startswith("gs://"):
        return None

    # Detect mime type
    ext = gcs_path.lower().split('.')[-1]
    mime_type = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'webp': 'image/webp'}.get(ext, 'image/png')

    if model_type == "gemini":
        return {'file_uri': gcs_path, 'media_type': mime_type}

    # Non-Gemini models need base64 data
    import base64
    image_bytes = download_to_bytes(gcs_path)
    if not image_bytes:
        return None
    return {'data': base64.b64encode(image_bytes).decode('utf-8'), 'media_type': mime_type}