"""Agent registry descriptions"""

# Agent registry for active agents in the system
AGENT_REGISTRY = {
    "script_agent": {
        "description": "Creates video scripts with character profiles (attributes, role) and scene planning",
        "capabilities": ["script_creation", "character_attributes_design", "story_development", "scene_planning", "dialogue_writing"],
        "system_prompt": "You are a video script writing specialist. Create detailed, engaging video scripts with character profiles (attributes and role), based on user requirements, creative direction, and production constraints."
    },
    "character_agent": {
        "description": "Generates character image visualizations (turnarounds and variations) based on script character profiles",
        "capabilities": ["character_image_generation", "visual_consistency", "turnaround_creation", "variation_generation"],
        "system_prompt": "You are a professional character visualization specialist. Generate character images (turnarounds and variations) that bring script character profiles to life visually."
    },
    "storyboard_agent": {
        "description": "Creates visual storyboards from scripts with character references for video production",
        "capabilities": ["storyboard_creation", "shot_composition", "visual_storytelling", "character_integration"],
        "system_prompt": "You are a professional storyboard artist. Create comprehensive storyboards that translate scripts into visual shots, ensuring proper framing, character consistency, and production readiness."
    },
    "supplementary_agent": {
        "description": "Flexible creative agent that handles any request not covered by specialized agents - generates concept art, mood boards, props, text content, and more",
        "capabilities": ["flexible_generation", "concept_art", "mood_boards", "props", "text_content", "metadata_creation"],
        "system_prompt": "You are a supplementary creative agent. Generate any type of content not covered by specialized agents - your output IS the metadata that will be stored and used by other agents."
    },
    "video_agent": {
        "description": "Creates hyper-specific AI video generation prompts for shot-to-shot transitions using storyboard frames",
        "capabilities": ["video_prompt_generation", "frame_analysis", "motion_description", "shot_transitions", "platform_optimization"],
        "system_prompt": "You are a professional AI video generation prompt specialist. Create hyper-specific prompts that describe exact frame-to-frame progressions for AI video generation models, ensuring visual continuity and platform-specific optimization."
    },
    "video_task_monitor": {
        "description": "Monitors video generation tasks submitted to AI video platforms and tracks their completion status",
        "capabilities": ["task_monitoring", "status_tracking", "video_download", "batch_monitoring"],
        "system_prompt": "You are a video task monitoring specialist. Track video generation tasks, monitor their progress, and report completion status."
    },
    "video_editor_agent": {
        "description": "Performs post-production video editing - combines clips, adds transitions, and overlays audio",
        "capabilities": ["video_editing", "clip_combination", "transitions", "audio_overlay"],
        "system_prompt": "You are a video editor specialist. Combine generated video clips in sequence, apply transitions, and add audio to create polished final videos."
    },
    "audio_agent": {
        "description": "Generates background music for video scenes based on script emotional beats and narrative pacing",
        "capabilities": ["music_generation", "scene_scoring", "mood_matching", "thematic_continuity"],
        "system_prompt": "You are a professional music supervisor and composer. Generate background music that enhances video storytelling, matches emotional beats, and maintains thematic continuity across scenes."
    }
}
