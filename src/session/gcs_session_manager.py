"""
Simple GCS storage for saved sessions - uploads/downloads entire folders
"""

import os
import json
import gzip
from pathlib import Path
from google.cloud import storage
from typing import List, Dict, Optional
from ..core.config import SESSION_BUCKET_NAME, CREDENTIALS, DEPLOYMENT_ENV


class GCSSessionStorage:
    """Simple GCS storage handler for saved sessions"""
    
    def __init__(self):
        """Initialize GCS storage client"""
        self.bucket_name = SESSION_BUCKET_NAME
        self.environment = DEPLOYMENT_ENV.lower() if DEPLOYMENT_ENV else 'local'
        
        try:
            self.storage_client = storage.Client(credentials=CREDENTIALS)
            self.bucket = self.storage_client.bucket(self.bucket_name)
            self.enabled = True
        except Exception as e:
            print(f"GCS storage not available: {e}")
            self.enabled = False
    
    def upload_session(self, local_session_path: Path, show_progress=None) -> Optional[str]:
        """
        Upload entire session folder to GCS
        
        Args:
            local_session_path: Path to local session folder
            show_progress: Optional callback for progress updates
            
        Returns:
            GCS path if successful, None otherwise
        """
        if not self.enabled:
            return None
            
        try:
            session_name = local_session_path.name
            gcs_prefix = f"{self.environment}/saved_sessions/{session_name}"
            
            # Count files for progress
            files_to_upload = list(local_session_path.rglob("*"))
            total_files = sum(1 for f in files_to_upload if f.is_file())
            uploaded = 0
            
            # Upload all files in session folder
            for file_path in files_to_upload:
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_session_path)
                    blob_name = f"{gcs_prefix}/{relative_path.as_posix()}"
                    
                    # Compress state.json to save bandwidth
                    if file_path.name == "state.json":
                        blob = self.bucket.blob(f"{blob_name}.gz")
                        with open(file_path, 'rb') as f:
                            compressed_data = gzip.compress(f.read())
                            blob.upload_from_string(
                                compressed_data,
                                content_type='application/gzip'
                            )
                    else:
                        blob = self.bucket.blob(blob_name)
                        blob.upload_from_filename(str(file_path))
                    
                    uploaded += 1
                    if show_progress:
                        show_progress(f"Uploaded {uploaded}/{total_files}: {relative_path.name}")
            
            return f"gs://{self.bucket_name}/{gcs_prefix}"
            
        except Exception as e:
            print(f"Error uploading session: {e}")
            return None
    
    def download_session(self, session_name: str, local_dir: Path) -> Optional[Path]:
        """
        Download entire session folder from GCS
        
        Args:
            session_name: Name of session to download
            local_dir: Local directory to download to
            
        Returns:
            Path to downloaded session if successful, None otherwise
        """
        if not self.enabled:
            return None
            
        try:
            gcs_prefix = f"{self.environment}/saved_sessions/{session_name}"
            local_session_path = local_dir / session_name
            
            # Create local directory
            local_session_path.mkdir(parents=True, exist_ok=True)
            
            # List and download all blobs
            blobs = list(self.bucket.list_blobs(prefix=gcs_prefix))
            if not blobs:
                print(f"No session found at {gcs_prefix}")
                return None
            
            for blob in blobs:
                # Get relative path after prefix
                if not blob.name.startswith(gcs_prefix + '/'):
                    continue
                    
                relative_path = blob.name[len(gcs_prefix)+1:]
                if not relative_path:
                    continue
                
                # Handle compressed state.json
                if relative_path.endswith("state.json.gz"):
                    local_file = local_session_path / "state.json"
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download and decompress
                    compressed_content = blob.download_as_bytes()
                    decompressed_content = gzip.decompress(compressed_content)
                    local_file.write_bytes(decompressed_content)
                else:
                    local_file = local_session_path / relative_path
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    blob.download_to_filename(str(local_file))
            
            return local_session_path
            
        except Exception as e:
            print(f"Error downloading session: {e}")
            return None
    
    def list_gcs_sessions(self) -> List[Dict]:
        """
        List all sessions in GCS for current environment
        
        Returns:
            List of session metadata dictionaries
        """
        if not self.enabled:
            return []
            
        try:
            prefix = f"{self.environment}/saved_sessions/"
            
            # Get unique session folders
            sessions = set()
            blobs = self.bucket.list_blobs(prefix=prefix, delimiter='/')
            
            # Must iterate through blobs to populate prefixes
            for _ in blobs:
                pass  # Just iterate to trigger prefix detection
            
            # Now process prefixes (folders)
            for prefix_name in blobs.prefixes:
                # Extract session name from path
                session_name = prefix_name[len(prefix):].rstrip('/')
                if session_name:
                    sessions.add(session_name)
            
            # Get metadata for each session
            session_list = []
            for session_name in sessions:
                try:
                    # Try to get metadata
                    metadata_blob = self.bucket.blob(f"{prefix}{session_name}/metadata.json")
                    if metadata_blob.exists():
                        metadata_content = metadata_blob.download_as_text()
                        metadata = json.loads(metadata_content)
                        
                        session_list.append({
                            "name": session_name,
                            "title": metadata.get("session_name", session_name),
                            "created": metadata.get("saved_at", metadata.get("created_at", "")),
                            "cloud": True,
                            "environment": self.environment,
                            "thread_id": metadata.get("original_thread_id", ""),
                            "message_count": metadata.get("message_count", 0),
                            "turn_count": metadata.get("turn_count", 0)
                        })
                except Exception as e:
                    # If metadata fails, still include session with basic info
                    session_list.append({
                        "name": session_name,
                        "title": session_name,
                        "created": "",
                        "cloud": True,
                        "environment": self.environment,
                        "thread_id": ""
                    })
            
            # Sort by created date (newest first)
            session_list.sort(
                key=lambda x: x.get("created", ""),
                reverse=True
            )
            
            return session_list
            
        except Exception as e:
            print(f"Error listing GCS sessions: {e}")
            return []
    
    def session_exists_in_gcs(self, session_name: str) -> bool:
        """
        Check if session exists in GCS
        
        Args:
            session_name: Name of session to check
            
        Returns:
            True if session exists, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            prefix = f"{self.environment}/saved_sessions/{session_name}/"
            # Check if any blobs exist with this prefix
            blobs = self.bucket.list_blobs(prefix=prefix, max_results=1)
            return any(blobs)
        except Exception as e:
            print(f"Error checking session existence: {e}")
            return False
    
    def delete_session(self, session_name: str) -> bool:
        """
        Delete a session from GCS
        
        Args:
            session_name: Name of session to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            prefix = f"{self.environment}/saved_sessions/{session_name}/"
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            # Delete all blobs with this prefix
            deleted_count = 0
            for blob in blobs:
                blob.delete()
                deleted_count += 1
            
            print(f"Deleted {deleted_count} files from GCS session {session_name}")
            return deleted_count > 0
            
        except Exception as e:
            print(f"Error deleting session from GCS: {e}")
            return False