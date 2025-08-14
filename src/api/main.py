"""Main FastAPI application with WebSocket support"""

import os
import uuid
import json
import asyncio
import tempfile
import shutil
import aiohttp
import mimetypes
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from urllib.parse import urlparse
from werkzeug.utils import secure_filename

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis

from .websocket import WebSocketHandler
from .api_types import SessionResponse, SessionRequest, SaveSessionRequest, UploadURLResponse, ImportReferencesRequest
from ..session.gcs_session_manager import GCSSessionStorage

# Redis connection pool
redis_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    global redis_pool
    
    # Use REDIS_URL_TEST for now (production will use REDIS_URL later)
    redis_url = os.getenv("REDIS_URL_TEST")
    if not redis_url:
        raise ValueError("[API] REDIS_URL_TEST must be set")
    
    redis_pool = redis.ConnectionPool.from_url(
        redis_url, 
        max_connections=30,  # Match redis_state.py settings
        decode_responses=True,
        socket_keepalive=True
    )
    app.state.redis = redis.Redis(connection_pool=redis_pool)
    # print(f"[API] Connected to Redis at {redis_url}")

    yield

    # Shutdown
    if app.state.redis:
        await app.state.redis.aclose()
        # print("[API] Redis connection closed")

# Initialize FastAPI app
app = FastAPI(
    title="Agent System Backend Service",
    description="Stateful backend for multi-agent workflow with real-time streaming",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for client applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure with actual client domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Agent System Backend",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Check Redis connection
        await app.state.redis.ping()
        redis_status = "connected"
    except Exception as e:
        redis_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "redis": redis_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: Optional[SessionRequest] = None):
    """Create a new workflow session"""
    # Use provided session_id or generate new one
    if request and request.session_id:
        session_id = request.session_id
    else:
        session_id = str(uuid.uuid4())
    
    # Initialize session metadata
    metadata = {}
    if request and request.metadata:
        metadata = request.metadata

    # Initialize session in Redis with 24hr TTL
    session_data = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "metadata": metadata,
        "state": {}
    }
    
    await app.state.redis.setex(
        f"session:{session_id}",
        86400,  # 24 hour TTL
        json.dumps(session_data)
    )

    # print(f"[API] Created new session: {session_id}")
    return SessionResponse(session_id=session_id, status="created")

@app.get("/sessions")
async def list_sessions():
    """List all sessions (both temporary and saved)"""
    all_sessions = []

    # Get temporary sessions from Redis
    session_keys = []
    cursor = 0

    while True:
        cursor, keys = await app.state.redis.scan(
            cursor,
            match="session:*",
            count=100
        )
        session_keys.extend(keys)
        if cursor == 0:
            break

    # Process Redis sessions
    for key in session_keys:
        session_data = await app.state.redis.get(key)
        if session_data:
            data = json.loads(session_data)
            ttl = await app.state.redis.ttl(key)
            data["ttl_seconds"] = ttl
            data["storage_type"] = "temporary"
            all_sessions.append(data)

    # Get saved sessions from GCS
    try:
        gcs_storage = GCSSessionStorage()
        if gcs_storage.enabled:
            saved_sessions = gcs_storage.list_gcs_sessions()
            for session in saved_sessions:
                # Convert GCS format to match Redis format
                all_sessions.append({
                    "session_id": session.get("name"),
                    "created_at": session.get("created", ""),
                    "storage_type": "saved",
                    "metadata": {
                        "title": session.get("title", ""),
                        "message_count": session.get("message_count", 0),
                        "turn_count": session.get("turn_count", 0)
                    }
                })
    except Exception as e:
        pass
        # print(f"[API] Error fetching saved sessions: {e}")

    # Sort by creation time (newest first)
    all_sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return {
        "sessions": all_sessions,
        "count": len(all_sessions)
    }

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session metadata and state"""
    session_key = f"session:{session_id}"
    session_data = await app.state.redis.get(session_key)
    
    if not session_data:
        return {"error": "Session not found", "session_id": session_id}
    
    # Get session metadata
    result = json.loads(session_data)
    
    # Also get the actual workflow state from state_hash
    state_key = f"state_hash:{session_id}"
    raw_state = await app.state.redis.hgetall(state_key)
    
    if raw_state:
        # Decode hash fields
        state = {}
        for field, value in raw_state.items():
            try:
                state[field] = json.loads(value)
            except json.JSONDecodeError:
                state[field] = value
        result["workflow_state"] = state
    else:
        result["workflow_state"] = {}
    
    return result

@app.get("/sessions/{session_id}/upload-urls", response_model=UploadURLResponse)
async def generate_upload_urls(
    session_id: str,
    count: int = 1,
    filenames: Optional[str] = None
):
    """
    Generate pre-signed URLs for direct GCS upload

    Args:
        session_id: Session identifier
        count: Number of upload URLs to generate (max 10)
        filenames: Comma-separated list of filenames (optional)

    Returns:
        Upload URLs for direct client-to-GCS upload + public URLs for workflow
    """
    # Validate session exists
    session_key = f"session:{session_id}"
    if not await app.state.redis.exists(session_key):
        return {"error": "Session not found", "session_id": session_id}

    # Initialize GCS
    from google.cloud import storage
    from ..core.config import SESSION_BUCKET_NAME, CREDENTIALS, DEPLOYMENT_ENV

    try:
        storage_client = storage.Client(credentials=CREDENTIALS)
        bucket = storage_client.bucket(SESSION_BUCKET_NAME)
    except Exception as e:
        return {"error": f"GCS initialization error: {str(e)}", "session_id": session_id}

    # Parse filenames or generate defaults
    if filenames:
        file_list = [f.strip() for f in filenames.split(',')]
    else:
        file_list = [f"reference_{i+1}.jpg" for i in range(count)]

    # Limit to 10 files
    file_list = file_list[:10]

    # Generate URLs
    urls = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    environment = DEPLOYMENT_ENV.lower() if DEPLOYMENT_ENV else 'local'

    for idx, filename in enumerate(file_list):
        # Sanitize filename
        safe_name = secure_filename(filename)
        if not safe_name:
            safe_name = f"reference_{idx+1}.jpg"

        # Build GCS path (match session folder structure)
        blob_name = f"{environment}/sessions/{session_id}/references/{timestamp}_{safe_name}"
        blob = bucket.blob(blob_name)

        # Generate upload URL (15 min for security)
        upload_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="PUT"
        )

        # Generate public URL (7 days, matches all assets)
        public_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(days=7),
            method="GET"
        )

        urls.append({
            "filename": safe_name,
            "upload_url": upload_url,
            "public_url": public_url,
            "gcs_path": f"gs://{SESSION_BUCKET_NAME}/{blob_name}"
        })

    return {
        "session_id": session_id,
        "urls": urls,
        "upload_instructions": {
            "method": "PUT",
            "note": "Upload file directly to upload_url, then use public_url in WebSocket messages",
            "expires_in_seconds": 900
        }
    }

@app.post("/sessions/{session_id}/upload")
async def upload_files(session_id: str, files: list[UploadFile] = File(...)):
    """
    Unified upload endpoint for images and documents

    Automatically detects file type and routes to appropriate storage:
    - Images (.png, .jpg, .jpeg, .webp, .gif) → references/ folder
    - Documents (.pdf, .ppt, .pptx, .doc, .docx) → documents/ folder

    Args:
        session_id: Session identifier
        files: List of image/document files (max 20)

    Returns:
        URLs, types, and counts for uploaded files
    """
    # Validate session exists
    session_key = f"session:{session_id}"
    if not await app.state.redis.exists(session_key):
        return {"error": "Session not found", "session_id": session_id}

    # Limit to 20 files
    if len(files) > 20:
        return {"error": "Maximum 20 files per upload", "session_id": session_id}

    from ..storage.gcs_utils import upload_bytes_to_gcs
    import asyncio
    import mimetypes

    # Supported types
    IMAGE_TYPES = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
    DOCUMENT_TYPES = {'.pdf', '.ppt', '.pptx', '.doc', '.docx'}

    async def upload_file(file: UploadFile, index: int) -> dict:
        """Upload single file with type detection"""
        try:
            file_data = await file.read()
            safe_name = secure_filename(file.filename) if file.filename else f"file_{index+1}"
            ext = Path(safe_name).suffix.lower()

            # Validate file type
            if ext not in IMAGE_TYPES and ext not in DOCUMENT_TYPES:
                return {"error": f"Unsupported file type: {ext} ({file.filename})"}

            # Determine type and folder
            if ext in IMAGE_TYPES:
                resource_type = "image"
                folder = "references"
            else:
                resource_type = "document"
                folder = "documents"

            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{safe_name}"

            # Upload to GCS - request gs:// path format
            public_url = await asyncio.to_thread(
                upload_bytes_to_gcs,
                file_data,
                session_id,
                folder,
                filename,
                file.content_type or mimetypes.guess_type(safe_name)[0],
                "gs"  # Request gs:// path format (50 chars vs 500+ for signed URL)
            )

            if not public_url:
                return {"error": f"Upload failed for {file.filename}"}

            return {
                "url": public_url,
                "filename": safe_name,
                "type": resource_type,
                "extension": ext
            }

        except Exception as e:
            return {"error": f"Upload failed for {file.filename}: {str(e)}"}

    # Upload files in parallel
    upload_tasks = [upload_file(file, i) for i, file in enumerate(files)]
    results = await asyncio.gather(*upload_tasks)

    # Organize results - single pass for efficiency
    image_urls = []
    document_urls = []
    documents = []  # Full document metadata for enterprise_resources
    errors = []

    for i, result in enumerate(results):
        if "error" in result:
            errors.append(result["error"])
        else:
            # print(f"[API Upload Debug] File {i}: type='{result.get('type')}', ext='{result.get('extension')}', filename='{result.get('filename')}'")

            if result["type"] == "image":
                image_urls.append(result["url"])
                # print(f"[API] → Classified as IMAGE: '{result.get('filename')}'")
            else:
                # Document - save URL and full metadata
                document_urls.append(result["url"])
                documents.append({
                    "url": result["url"],
                    "filename": result["filename"],
                    "extension": result["extension"],
                    "timestamp": datetime.now().isoformat()
                })
                # print(f"[API] → Classified as DOCUMENT: '{result.get('filename')}' (ext: {result.get('extension')})")

    # Update state with new uploads
    if image_urls or document_urls:
        try:
            state_key = f"state_hash:{session_id}"

            # Update reference_images (for images ONLY)
            if image_urls:
                current_refs_json = await app.state.redis.hget(state_key, "reference_images")
                reference_images = json.loads(current_refs_json) if current_refs_json else []
                reference_images.extend(image_urls)
                await app.state.redis.hset(state_key, "reference_images", json.dumps(reference_images))
                # print(f"[API] ✓ Added {len(image_urls)} image(s) to reference_images")

            # Update enterprise_resources (for documents ONLY)
            if documents:
                current_resources_json = await app.state.redis.hget(state_key, "enterprise_resources")
                enterprise_resources = json.loads(current_resources_json) if current_resources_json else {"images": [], "documents": []}

                # print(f"[API] Before update: enterprise_resources = {enterprise_resources}")
                enterprise_resources["documents"].extend(documents)
                # print(f"[API] After extend: enterprise_resources = {enterprise_resources}")

                await app.state.redis.hset(state_key, "enterprise_resources", json.dumps(enterprise_resources))
                # print(f"[API] ✓ Saved to Redis key '{state_key}' field 'enterprise_resources'")
                # print(f"[API] ✓ Added {len(documents)} document(s) to enterprise_resources")

                # Verify it was saved
                verify_json = await app.state.redis.hget(state_key, "enterprise_resources")
                verify_data = json.loads(verify_json) if verify_json else None
                # print(f"[API] ✓ Verification read from Redis: {verify_data}")

        except Exception as e:
            pass
            # print(f"[API] Warning: Could not update state: {e}")

    # Return gs:// paths directly - client will resolve to signed URLs on-demand
    # print(f"[API Upload] Returning {len(image_urls) + len(document_urls)} gs:// paths (client resolves URLs on-demand)")

    return {
        "session_id": session_id,
        "image_urls": image_urls,  # Return gs:// paths
        "document_urls": document_urls,  # Return gs:// paths
        "errors": errors if errors else None,
        "image_count": len(image_urls),
        "document_count": len(document_urls),
        "total_count": len(image_urls) + len(document_urls)
    }

@app.post("/sessions/{session_id}/enterprise/upload-resources")
async def upload_enterprise_resources(session_id: str, files: list[UploadFile] = File(...)):
    """
    DEPRECATED: Use POST /sessions/{session_id}/upload instead

    This endpoint redirects to the unified upload endpoint for backward compatibility
    """
    # Redirect to unified upload
    return await upload_files(session_id, files)

@app.post("/sessions/{session_id}/import-references")
async def import_reference_urls(
    session_id: str,
    request: ImportReferencesRequest
):
    """
    Import external image URLs by downloading and uploading to GCS

    Args:
        session_id: Session identifier
        request: ImportReferencesRequest with urls and optional filenames

    Returns:
        Public URLs for imported images
    """
    # Validate session exists
    session_key = f"session:{session_id}"
    if not await app.state.redis.exists(session_key):
        return {"error": "Session not found", "session_id": session_id}

    urls = request.urls
    filenames = request.filenames or []

    if not urls:
        return {"error": "No URLs provided", "session_id": session_id}

    if len(urls) > 10:
        return {"error": "Maximum 10 images per import", "session_id": session_id}

    # Import GCS utilities
    from ..storage.gcs_utils import upload_bytes_to_gcs

    async def download_and_upload(url: str, index: int) -> dict:
        """Download image from URL and upload to GCS"""
        try:
            # Use provided filename or extract from URL
            if index < len(filenames) and filenames[index]:
                safe_name = secure_filename(filenames[index])
            else:
                parsed = urlparse(url)
                path_name = parsed.path.split('/')[-1] or f"reference_{index+1}.jpg"
                safe_name = secure_filename(path_name)

            # Download image
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        return {"error": f"Error downloading from {url}: HTTP {response.status}"}

                    image_data = await response.read()

                    # Detect content type
                    content_type = response.headers.get('Content-Type')
                    if not content_type or not content_type.startswith('image/'):
                        content_type = mimetypes.guess_type(safe_name)[0] or 'image/jpeg'

            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{safe_name}"

            # Upload to GCS - request gs:// path format (runs in thread pool)
            public_url = await asyncio.to_thread(
                upload_bytes_to_gcs,
                image_data,
                session_id,
                "references",
                filename,
                content_type,
                "gs"  # Request gs:// path format
            )

            if not public_url:
                return {"error": f"GCS upload error for {url}"}

            return {"url": public_url, "filename": safe_name, "source_url": url}

        except asyncio.TimeoutError:
            return {"error": f"Download timeout for {url}"}
        except Exception as e:
            return {"error": f"Import failed for {url}: {str(e)}"}

    # Process imports in parallel
    import_tasks = [download_and_upload(url, i) for i, url in enumerate(urls)]
    results = await asyncio.gather(*import_tasks)

    # Extract URLs and errors
    imported_urls = []
    errors = []
    for result in results:
        if "error" in result:
            errors.append(result["error"])
        else:
            imported_urls.append(result["url"])

    # Return gs:// paths directly - client will resolve to signed URLs on-demand
    # if imported_urls:
    #     print(f"[API Import] Returning {len(imported_urls)} gs:// paths (client resolves URLs on-demand)")

    return {
        "session_id": session_id,
        "urls": imported_urls,  # Return gs:// paths
        "errors": errors if errors else None,
        "imported_count": len(imported_urls)
    }

@app.post("/sessions/{session_id}/resolve-urls")
async def resolve_gcs_urls(session_id: str, request: dict):
    """
    Batch convert gs:// paths to signed URLs for client display

    This endpoint allows the frontend to convert gs:// paths to browser-displayable
    signed URLs on-demand with caching support.

    Args:
        session_id: Session identifier
        request: {"paths": ["gs://...", "gs://..."]}

    Returns:
        {
            "urls": {"gs://path": "https://signed-url", ...},
            "expiry": "ISO timestamp string",
            "count": 5
        }
    """
    from ..storage.gcs_utils import generate_signed_url
    from datetime import datetime, timedelta

    paths = request.get("paths", [])

    if not paths:
        return {"error": "No paths provided", "urls": {}, "count": 0}

    # print(f"[API Resolve URLs] Converting {len(paths)} gs:// paths to signed URLs...")
    # print(f"[API Resolve URLs] Supports all asset types: images, videos, audio, documents, references")

    urls = {}
    for path in paths:
        if path and path.startswith("gs://"):
            signed_url = generate_signed_url(path, expiration_days=7)
            if signed_url:
                urls[path] = signed_url
                # print(f"[API Resolve URLs]   ✓ {path.split('/')[-1]}")
            # else:
                # print(f"[API Resolve URLs]   ✗ Failed: {path}")

    # Return expiry time so client knows when cache expires
    expiry_time = (datetime.now() + timedelta(days=7)).isoformat()

    # print(f"[API Resolve URLs] Successfully converted {len(urls)}/{len(paths)} URLs (7-day expiry)")

    return {
        "urls": urls,
        "expiry": expiry_time,
        "count": len(urls)
    }

@app.get("/sessions/{session_id}/export")
async def export_session(session_id: str):
    """
    Export session state for client applications

    Returns internal state with gs:// paths. Client should use
    POST /sessions/{id}/resolve-urls to convert paths to signed URLs on-demand.

    Args:
        session_id: Session identifier
    """
    # Get state from Redis hash
    state_key = f"state_hash:{session_id}"
    raw_state = await app.state.redis.hgetall(state_key)

    if not raw_state:
        return {"error": "Session not found", "session_id": session_id}

    # Decode hash fields
    state = {}
    for field, value in raw_state.items():
        try:
            state[field] = json.loads(value)
        except json.JSONDecodeError:
            state[field] = value

    # Always return internal state with gs:// paths
    # Client will use /resolve-urls endpoint to convert paths on-demand
    # print(f"[API Export] Returning internal state with gs:// paths (client resolves URLs on-demand)")

    expert_states = {}

    # Helper function to extract URLs from function_calls
    # Use utility function from base.py for consistent metadata access
    from ..agents.base import get_all_asset_urls

    all_assets = get_all_asset_urls(state)

    # Extract all assets more cleanly
    assets = {
        "user_references": all_assets.get("references", []),
        "character_images": all_assets.get("characters", []),
        "storyboard_frames": all_assets.get("storyboards", []),
        "supplementary_assets": all_assets.get("supplementary", []),
        "generated_videos": all_assets.get("videos", []),
        "edited_videos": [video.get("path") for video in state.get("edited_videos", []) if video.get("path")],
        "generated_audio": all_assets.get("audio", [])
    }
    
    # Package core content for export
    from datetime import datetime
    export_data = {
        "session_id": session_id,
        "exported_at": datetime.now().isoformat(),
        "assets": assets,
        "generated_scripts": state.get("generated_scripts", {}),
        "character_image_metadata": state.get("character_image_metadata", {}),
        "storyboard_frame_metadata": state.get("storyboard_frame_metadata", {}),
        "video_generation_metadata": state.get("video_generation_metadata", {}),
        "audio_generation_metadata": state.get("audio_generation_metadata", {}),
        "generated_storyboards": state.get("generated_storyboards", {}),
        "generated_video_prompts": state.get("generated_video_prompts", {}),
        "generated_supplementary": state.get("generated_supplementary", {}),
        "expert_states": expert_states,
        "annotations": state.get("image_annotations", {})
    }
    
    return export_data

@app.get("/sessions/saved")
async def list_saved_sessions():
    """List all saved sessions from GCS"""
    try:
        gcs_storage = GCSSessionStorage()
        if not gcs_storage.enabled:
            return {"error": "GCS storage not available", "sessions": []}

        saved_sessions = gcs_storage.list_gcs_sessions()
        return {
            "sessions": saved_sessions,
            "count": len(saved_sessions)
        }
    except Exception as e:
        return {"error": f"Failed to list saved sessions: {str(e)}", "sessions": []}

@app.post("/sessions/{session_id}/restore")
async def restore_session(session_id: str):
    """Restore session from GCS to Redis"""
    try:
        gcs_storage = GCSSessionStorage()
        if not gcs_storage.enabled:
            return {"error": "GCS storage not available", "session_id": session_id}

        # Download session from GCS to temporary directory
        temp_dir = Path(tempfile.mkdtemp())

        session_path = gcs_storage.download_session(session_id, temp_dir)
        if not session_path:
            shutil.rmtree(temp_dir)
            return {"error": "Session not found in GCS", "session_id": session_id}

        # Load state from downloaded files
        state_file = session_path / "state.json"
        if not state_file.exists():
            shutil.rmtree(temp_dir)
            return {"error": "Session state file not found", "session_id": session_id}

        state = json.loads(state_file.read_text())

        # Load metadata
        metadata = {}
        metadata_file = session_path / "metadata.json"
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())

        # Remove legacy expert states if present in saved session
        state.pop('_expert_states', None)

        # Regenerate fresh signed URLs for all assets
        from ..storage.gcs_utils import regenerate_session_urls
        state = regenerate_session_urls(state, session_id)

        # Save to Redis
        state_key = f"state_hash:{session_id}"

        # Convert state to Redis hash format
        encoded_state = {}
        for field, value in state.items():
            if value is not None:
                encoded_state[field] = json.dumps(value, default=str)

        # Use pipeline for atomic operation
        pipe = app.state.redis.pipeline()
        pipe.delete(state_key)
        if encoded_state:
            pipe.hset(state_key, mapping=encoded_state)
            pipe.expire(state_key, 86400)  # 24 hour TTL

        # Create session metadata entry
        session_data = {
            "session_id": session_id,
            "created_at": metadata.get("saved_at", datetime.now().isoformat()),
            "metadata": metadata,
            "restored_at": datetime.now().isoformat(),
            "state": {}
        }
        pipe.setex(
            f"session:{session_id}",
            86400,  # 24 hour TTL
            json.dumps(session_data)
        )

        await pipe.execute()

        # Clean up temp directory
        shutil.rmtree(temp_dir)

        return {
            "status": "restored",
            "session_id": session_id,
            "restored_at": session_data["restored_at"]
        }

    except Exception as e:
        return {"error": f"Failed to restore session: {str(e)}", "session_id": session_id}

@app.post("/sessions/{session_id}/save")
async def save_session(session_id: str, request: SaveSessionRequest = SaveSessionRequest()):
    """Save session permanently to GCS with optional custom name"""
    # Get session data from Redis
    state_key = f"state_hash:{session_id}"
    raw_state = await app.state.redis.hgetall(state_key)

    if not raw_state:
        return {"error": "Session not found", "session_id": session_id}

    # Decode state
    state = {}
    for field, value in raw_state.items():
        try:
            state[field] = json.loads(value)
        except json.JSONDecodeError:
            state[field] = value

    # Use provided session_name or fall back to session_id
    save_name = request.session_name if request.session_name else session_id

    # Copy assets to permanent location and remap URLs
    from ..storage.gcs_utils import copy_session_assets, remap_asset_urls

    copied_count, filename_mapping = copy_session_assets(session_id, save_name)
    # print(f"[API] Copied {copied_count} assets for session {save_name}")

    state = remap_asset_urls(state, filename_mapping)

    # Create temporary directory for session export
    temp_dir = Path(tempfile.mkdtemp())
    session_dir = temp_dir / save_name
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save state.json
        state_file = session_dir / "state.json"
        state_file.write_text(json.dumps(state, indent=2, default=str))

        # Save metadata
        metadata = {
            "session_id": session_id,
            "session_name": save_name,
            "saved_at": datetime.now().isoformat(),
            "message_count": len(state.get("messages", [])),
            "turn_count": state.get("turn_count", 0)
        }
        metadata_file = session_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

        # Upload to GCS
        gcs_storage = GCSSessionStorage()
        if not gcs_storage.enabled:
            return {"error": "GCS storage not available", "session_id": session_id}

        gcs_path = gcs_storage.upload_session(session_dir)

        # Clean up temp directory
        shutil.rmtree(temp_dir)

        if gcs_path:
            return {
                "status": "saved",
                "session_id": session_id,
                "session_name": save_name,
                "gcs_path": gcs_path,
                "saved_at": metadata["saved_at"]
            }
        else:
            return {"error": "Error uploading to GCS", "session_id": session_id}

    except Exception as e:
        # Clean up on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return {"error": f"Failed to save session: {str(e)}", "session_id": session_id}

# REMOVED: Enterprise query endpoint - enterprise analysis now runs automatically in memory_updater
# Documents are analyzed once when uploaded, output available in state["enterprise_agent_output"]

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session from both Redis and GCS"""
    deleted_from = []

    # Delete from Redis
    keys_to_delete = [
        f"session:{session_id}",
        f"state:{session_id}",
        f"state_hash:{session_id}",
        f"events:{session_id}",
        f"messages:{session_id}",
        f"history:{session_id}"
    ]

    existing_keys = []
    for key in keys_to_delete:
        if await app.state.redis.exists(key):
            existing_keys.append(key)

    if existing_keys:
        await app.state.redis.delete(*existing_keys)
        deleted_from.append("redis")
        # print(f"[API] Deleted session {session_id} from Redis ({len(existing_keys)} keys)")

    # Delete from GCS
    try:
        gcs_storage = GCSSessionStorage()
        if gcs_storage.enabled:
            if gcs_storage.session_exists_in_gcs(session_id):
                if gcs_storage.delete_session(session_id):
                    deleted_from.append("gcs")
                    # print(f"[API] Deleted session {session_id} from GCS")
    except Exception as e:
        pass
        # print(f"[API] Error deleting session from GCS: {e}")

    if deleted_from:
        return {
            "status": "deleted",
            "session_id": session_id,
            "deleted_from": deleted_from
        }
    else:
        return {"status": "not_found", "session_id": session_id}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time streaming"""
    
    # Validate session exists before accepting connection
    session_key = f"session:{session_id}"
    session_exists = await app.state.redis.exists(session_key)

    if not session_exists:
        # print(f"[WebSocket] Rejected connection - session {session_id} not found")
        await websocket.accept()
        await websocket.close(code=1008, reason="Session not found")
        return

    await websocket.accept()
    # print(f"[WebSocket] Client connected to session {session_id}")
    
    # Update session last_connected timestamp
    session_data = await app.state.redis.get(session_key)
    if session_data:
        data = json.loads(session_data)
        data["last_connected"] = datetime.now().isoformat()
        await app.state.redis.setex(
            session_key,
            86400,  # Refresh TTL on connect (24 hours)
            json.dumps(data)
        )
    
    handler = WebSocketHandler(
        websocket=websocket,
        session_id=session_id,
        redis_client=app.state.redis
    )
    
    try:
        await handler.handle()
    except WebSocketDisconnect:
        pass
        # print(f"[WebSocket] Client disconnected from session {session_id}")
    except Exception as e:
        import traceback
        # print(f"[WebSocket] Error in session {session_id}: {str(e)}")
        # print(f"[WebSocket] Traceback:\n{traceback.format_exc()}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass  # WebSocket already closed

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)