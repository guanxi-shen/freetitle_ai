"""Configuration and setup for AI Video Studio workflow"""

import os
import json
from dotenv import load_dotenv
import vertexai
from google.oauth2 import service_account

# Load environment variables
load_dotenv()

# GCP Configuration
PROJECT_ID = os.getenv('GCP_PROJECT_ID')
LOCATION = os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')

# Storage Configuration
BUCKET_NAME = os.getenv('BUCKET_NAME')
SESSION_BUCKET_NAME = os.getenv('SESSION_BUCKET_NAME', '')
PUBLIC_BUCKET_NAME = os.getenv('PUBLIC_BUCKET_NAME', '')

# Memory Configuration
ROLLING_WINDOW_SIZE = 10
CONTEXT_SUMMARY_THRESHOLD = ROLLING_WINDOW_SIZE

# Retry Configuration
DEFAULT_MAX_RETRIES = 3

# Debug Configuration
DEBUG_TOKEN_COUNTING = False  # Enable detailed token counting per LLM call
DEBUG_CONTEXT = True  # Capture and export full prompts sent to LLMs

# Script Agent Document Access
SCRIPT_AGENT_USE_RAW_DOCUMENTS = True  # Enable direct PDF/document access for script agent

# Cinematic Style Directives
ENABLE_CINEMATIC_STYLE = True  # Append cinematic style directives to script agent prompt

# Context Caching Configuration
ENABLE_CACHE = False  # Enable/disable context caching

# Video Production Configuration
DEFAULT_VIDEO_DURATION = 60  # seconds
SCRIPT_FORMAT_VERSION = "1.0"
MAX_SCENES_PER_SCRIPT = 20
DEFAULT_SCRIPT_STYLE = "narrative"

# MCP Server Configuration
DEPLOYMENT_ENV = os.getenv('DEPLOYMENT_ENV', 'DEV')
MCP_SERVER_URL = os.getenv('MCP_SERVER_URL', '')

MCP_REQUEST_TIMEOUT = 10.0  # seconds

# Gemini API Aspect Ratio Mapping
# Maps our preset names to Gemini's aspect ratio format
# Also accepts Gemini formats directly for defensive programming
GEMINI_ASPECT_RATIOS = {
    # Preset names (preferred)
    "vertical": "9:16",
    "horizontal": "16:9",
    "square": "1:1",
    "cinematic": "21:9",
    "portrait": "3:4",
    "landscape": "4:3",
    # Gemini formats (defensive mapping - accept both formats)
    "9:16": "9:16",
    "16:9": "16:9",
    "1:1": "1:1",
    "21:9": "21:9",
    "3:4": "3:4",
    "4:3": "4:3"
}

# Production Pipeline Configuration
PRODUCTION_PIPELINE = {
    "pre_production": {
        "label": "PRE-PRODUCTION",
        "components": ["script", "characters", "storyboard", "casting"]
    },
    "production": {
        "label": "PRODUCTION", 
        "components": ["video", "audio", "music", "voice"]
    },
    "post_production": {
        "label": "POST-PRODUCTION",
        "components": ["editing", "color", "effects", "final"]
    }
}

# Component display names
COMPONENT_NAMES = {
    "script": "Script", 
    "characters": "Characters", 
    "storyboard": "Storyboard", 
    "casting": "Casting",
    "video": "Video", 
    "audio": "Audio", 
    "music": "Music", 
    "voice": "Voice",
    "editing": "Edit", 
    "color": "Color", 
    "effects": "Effects", 
    "final": "Final"
}

# Google Veo Configuration
GOOGLE_VEO_CONFIG = {
    "default_model": "veo-3.1-generate-preview",
    "max_wait_time": 600,
    "check_interval": 10,  # 10s polling per official examples
    "aspect_ratios": {
        "vertical": "9:16",
        "horizontal": "16:9"
    },
    "durations": [4, 6, 8],
    "resolutions": ["720p", "1080p", "4k"]  # Veo 3.1 supports up to 4K
}

# ElevenLabs API Configuration
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', '')
ELEVENLABS_API_ENDPOINT = "https://api.elevenlabs.io/v1"

# Video Generation Settings
VIDEO_GENERATION_CONFIG = {
    "max_wait_time": 600,  # 10 minutes
    "check_interval": 8,   # Check every 8 seconds
    "default_duration": 5,   # 5 seconds
    "auto_download": True,
    "aspect_ratios": {
        "vertical": "9:16",
        "horizontal": "16:9"
    }
}


# Music Generation Settings (ElevenLabs)
MUSIC_GENERATION_CONFIG = {
    "default_duration_ms": 60000,  # 60 seconds
    "default_output_format": "mp3_44100_128",  # Good quality, no subscription required
    "force_instrumental": True,  # Pure background music by default
    "max_duration_ms": 180000,  # 3 minutes maximum
    "output_formats": {
        "high_quality": "mp3_44100_192",  # Requires Creator tier+
        "standard": "mp3_44100_128",  # Good quality, free tier
        "low_quality": "mp3_22050_32"  # Smaller files, lower quality
    }
}

# Storyboard Validation Configuration
ENABLE_STORYBOARD_VALIDATION = False  # Enable visual quality validation for storyboards

# Storyboard Aspect Ratio Correction
ENABLE_ASPECT_RATIO_CORRECTION = True  # Crop storyboard frames to exact 16:9 or 9:16 for video generation

# Multimodal Function Response Configuration
INCLUDE_REFERENCE_IMAGES_IN_MULTIMODAL = True  # Include reference images in multimodal function responses for quality comparison
ENABLE_MULTIMODAL_VALIDATION = True  # Enable multimodal image validation with detailed instructions for character, supplementary, and storyboard agents

# Initialize Vertex AI and credentials
def initialize_vertex_ai():
    """Initialize Vertex AI with service account credentials for video production models"""
    credentials_json = os.getenv('credentials_dict')
    credentials_info = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    return credentials

# Initialize on module import
CREDENTIALS = initialize_vertex_ai()