"""Type definitions for API request/response validation"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

class SessionRequest(BaseModel):
    """Request type for creating a session"""
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
class SessionResponse(BaseModel):
    """Response type for session creation"""
    session_id: str
    status: str

class SaveSessionRequest(BaseModel):
    """Request type for saving a session"""
    session_name: Optional[str] = None

class WorkflowRequest(BaseModel):
    """Request type for workflow execution"""
    query: str
    context: Optional[Dict[str, Any]] = None
    reference_images: Optional[List[str]] = None
    
class StreamEvent(BaseModel):
    """Type for streaming events"""
    type: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
    agent: Optional[str] = None
    content: Optional[str] = None
    
class WebSocketMessage(BaseModel):
    """Type for WebSocket messages"""
    type: str
    query: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    reference_images: Optional[List[str]] = None

class ErrorResponse(BaseModel):
    """Type for error responses"""
    error: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class UploadURLItem(BaseModel):
    """Individual upload URL data"""
    filename: str
    upload_url: str
    public_url: str
    gcs_path: str

class UploadURLResponse(BaseModel):
    """Response with pre-signed upload URLs"""
    session_id: str
    urls: List[UploadURLItem]
    upload_instructions: Dict[str, Any]

class ImportReferencesRequest(BaseModel):
    """Request for importing external image URLs"""
    urls: List[str]
    filenames: Optional[List[str]] = None