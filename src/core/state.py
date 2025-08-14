"""State definition for Video Generation Workflow"""

from typing import List, Dict, Any, TypedDict, Union

class RAGState(TypedDict):
    # Current interaction
    user_query: str
    messages: List[Dict[str, Any]]  # Simple dicts with "role" and "content" keys

    # Orchestrator planning and routing
    execution_plan: str
    selected_agents: List[Union[str, List[str]]]  # Supports nested lists for parallel execution
    agent_instructions: Dict[str, str]
    
    # Video production fields (requirements extracted from user query and context)
    
    # Agent outputs
    script_output: Dict[str, Any]  # Generated video script
    character_output: Dict[str, Any]  # Character designs with flexible attributes
    storyboard_output: Dict[str, Any]  # Storyboard generation output
    audio_output: Dict[str, Any]  # Audio generation output
    video_output: Dict[str, Any]  # Video generation prompts output
    video_monitor_output: Dict[str, Any]  # Video task monitoring results
    storyboard_execution_output: Dict[str, Any]  # Storyboard execution results (turn-based)
    supplementary_monitor_output: Dict[str, Any]  # Supplementary task monitoring results
    video_editor_output: Dict[str, Any]  # Video editing results
    supplementary_output: Dict[str, Any]  # Supplementary agent output
    reference_expert_output: Dict[str, Any]  # Reference expert agent output

    # Character image generation prompts
    character_image_prompts: List[Dict[str, Any]]  # Extracted prompts for image generation
    
    # Storyboard image generation prompts
    storyboard_image_prompts: List[Dict[str, Any]]  # Extracted prompts for storyboard frames
    
    # Reference images and metadata (Redis - small data)
    reference_images: List[str]  # GCS URLs of user-uploaded reference images
    reference_usage_map: Dict[str, List[str]]  # Track which refs used where {"characters/Shinji": [urls]}
    image_annotations: Dict[str, Any]  # AI-generated descriptions {url: {description, filename, timestamp}}
    character_image_metadata: Dict[str, Any]  # Metadata for generated images {filename: {character_name, type, etc}}
    storyboard_frame_metadata: Dict[str, Any]  # Metadata for generated frames {filename: {scene, shot, frame, path, etc}}
    storyboard_validation_results: Dict[str, Any]  # Validation results for storyboard frames
    supplementary_content_metadata: Dict[str, Any]  # Metadata for supplementary content {key: {content_type, path, etc}}
    audio_generation_metadata: Dict[str, Any]  # Metadata for generated music {filename: {name, prompt, duration_ms, path, etc}}
    video_generation_metadata: Dict[str, Any]  # Metadata for generated videos {filename: {scene, shot, version, tool_used, params, metadata, etc}}

    # Intelligent video editing fields
    video_analysis_metadata: Dict[str, Dict[str, Any]]  # Video content analysis {filename: {duration, motion_intensity, peaks, trim_recommendation, etc}}
    audio_beat_analysis: Dict[str, Dict[str, Any]]  # Audio beat analysis {track_name: {tempo_bpm, beat_grid, downbeats, energy_peaks, etc}}
    
    # Persistent production assets - single source of truth (current versions only)
    generated_scripts: Dict[str, Any]  # Current generated script (includes character profiles)
    generated_storyboards: Dict[str, Any]  # Current generated storyboards
    generated_video_prompts: Dict[str, Any]  # Current generated video prompts with nested structure:
    # {
    #   "video_prompts": [
    #     {
    #       "scene_number": 1,
    #       "shot_number": 1,
    #       "start_frame_path": "gs://...",
    #       "end_frame_path": "gs://..." (optional),
    #       "generation_prompt": {
    #         "camera": "Camera angle and movement (10-15 words)",
    #         "motion": "Subject actions and transformations (20-30 words)",
    #         "style": "Visual aesthetic and quality (10-15 words)",
    #         "dialogue": "Exact dialogue with timing (if any)",
    #         "sound": "Key audio elements (10-15 words)",
    #         "note": "Consistency/logo requirements (10-20 words, optional)",
    #         "negative": "Things to avoid (10-15 words)"
    #       }
    #     }
    #   ],
    #   "metadata": {"total_shots": int, "timestamp": str}
    # }
    generated_supplementary: Dict[str, Any]  # Current generated supplementary content from supplementary_agent
    video_generation_tasks: List[Dict[str, Any]]  # Video generation task tracking
    edited_videos: List[Dict[str, Any]]  # Edited video outputs from video_editor_agent
    pinterest_references: Dict[str, Any]  # Pinterest visual references organized by category

    # TikTok expert workflow fields
    tiktok_decision: str  # TikTok orchestrator decision: "ignore" / "direct_response" / "research"
    tiktok_direct_answer: str  # Direct answer from orchestrator
    tiktok_reasoning: str  # Reasoning for direct answer
    tiktok_ignore_reason: str  # Reason for ignoring query
    tiktok_tools_requested: List[str]  # Tools requested for research
    tiktok_research_tasks: List[Dict[str, Any]]  # TikTok research tasks submitted to Apify
    tiktok_research_data: Dict[str, Any]  # TikTok research results from monitor
    tiktok_research_plan: str  # TikTok expert research plan
    tiktok_monitor_output: Dict[str, Any]  # TikTok monitor execution results
    tiktok_final_answer: str  # TikTok answer parser final response
    tiktok_monitor_timeout: int  # TikTok monitor timeout override

    # Enterprise document analysis fields
    enterprise_resources: Dict[str, List[Dict[str, Any]]]  # {
    #   "images": [{"url": "...", "filename": "...", "timestamp": "..."}],
    #   "documents": [{"url": "...", "filename": "...", "extension": ".pdf", "timestamp": "..."}]
    # }
    enterprise_agent_output: Dict[str, Any]  # One-time analysis output {
    #   "status": "success|error|no_resources",
    #   "analysis": "Natural text creative brief with business context, products, audience, creative opportunities...",
    #   "metadata": {"analyzed_files": [...], "timestamp": "...", "model": "gemini-2.5-pro-preview"}
    # }

    # Final response
    final_answer: str
    
    # Project memory
    thread_id: str
    session_id: str  # Session identifier for API/GCS operations
    # Asset URLs organized by source/type with metadata
    # Character images accessed via character_image_metadata (transformed on-demand by consumers)
    asset_urls: Dict[str, List[Dict[str, Any]]]  # {
        # "user_references": [{"url": "...", "filename": "...", "timestamp": "..."}],
        # "storyboard_frames": [{"url": "...", "scene": 1, "shot": 1, "frame": 1}],
        # "supplementary_assets": [{"url": "...", "type": "...", "name": "..."}],
        # "generated_videos": [{"url": "...", "scene": 1, "shot": 1, "version": 1}],
        # "edited_videos": [{"url": "...", "type": "combined", "timestamp": "..."}]
    # }
    conversation_history: List[Dict[str, Any]]
    window_memory: List[Dict[str, Any]]
    context_summary: str
    turn_number: int
    
    # Video project metadata
    user_preferences: Dict[str, Any]  # Contains tool_selections: {agent_name: [tool_ids]}, workflow_mode: str, enable_validation: bool
    expert_selections: List[str]  # Active expert agents selected by user
    metadata: Dict[str, Any]
    timestamp: str
    
    # Thinking processes for user display
    thinking_processes: Dict[str, str]

    # Debug prompt capture
    debug_prompts: Dict[str, Any]

    # Timing data
    component_timings: Dict[str, float]
    total_execution_time: float
    
    # Re-act cycle state for task completion feedback
    retry_count: int
    max_retries: int
    task_completion: Dict[str, Any]
    retry_rationale: str
    additional_assistance_needed: str
    previous_attempts: List[Dict[str, Any]]
    orchestrator_retry_context: Dict[str, Any]
    
    # Production pipeline state tracking
    production_pipeline_state: Dict[str, bool]
    
    # Current agent tracking for WebSocket events
    current_agent: str

    # Expert agent fields
    expert_id: str  # Expert identifier for event routing
    execution_id: str  # Execution tracking ID for expert workflows
    expert_messages: List[Dict[str, Any]]  # Array of all expert messages across session
    expert_conversation_history: Dict[str, List[Dict[str, Any]]]  # Per-expert conversation tracking with global turns
    expert_window_memory: Dict[str, List[Dict[str, Any]]]  # Recent expert conversations windowed by global turn
    reference_orchestrator_decision: Dict[str, Any]  # Reference expert orchestrator decision
    pending_confirmation: Dict[str, Any]  # Pending expert confirmation details