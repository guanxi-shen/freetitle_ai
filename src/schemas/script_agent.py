"""
Schema definitions for script_agent.
Supports both Gemini and GPT structured output.

Key difference: Gemini includes propertyOrdering (non-standard),
GPT has strict additionalProperties: false on all nested objects.
"""

GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "characters": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "attributes": {"type": "STRING"},
                    "role": {"type": "STRING"}
                },
                "required": ["name", "attributes", "role"]
            }
        },
        "script_details": {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING"},
                "duration": {"type": "STRING"},
                "video_summary": {"type": "STRING"},
                "creative_vision": {"type": "STRING"},
                "aspect_ratio": {"type": "STRING"},
                "scenes": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "scene_number": {"type": "INTEGER"},
                            "scene_summary": {"type": "STRING"},
                            "setting": {"type": "STRING"},
                            "duration": {"type": "STRING"},
                            "characters": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"}
                            },
                            "consistency_notes": {"type": "STRING"},
                            "shots": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "shot_number": {"type": "INTEGER"},
                                        "shot_type": {"type": "STRING"},
                                        "duration": {"type": "STRING"},
                                        "subject": {"type": "STRING"},
                                        "description": {"type": "STRING"},
                                        "shot_purpose": {"type": "STRING"},
                                        "start_frame": {"type": "STRING"},
                                        "end_frame": {"type": "STRING"},
                                        "progression": {"type": "STRING"},
                                        "visual_reference": {
                                            "type": "ARRAY",
                                            "items": {
                                                "type": "OBJECT",
                                                "properties": {
                                                    "shot_id": {"type": "STRING"},
                                                    "description": {"type": "STRING"}
                                                },
                                                "required": ["shot_id", "description"]
                                            }
                                        },
                                        "continuity_notes": {
                                            "type": "OBJECT",
                                            "properties": {
                                                "from_previous": {"type": "STRING"},
                                                "to_next": {"type": "STRING"},
                                                "transition_suggestion": {"type": "STRING"},
                                                "editing_intent": {"type": "STRING"}
                                            }
                                        },
                                        "key_visual_elements": {
                                            "type": "ARRAY",
                                            "items": {"type": "STRING"}
                                        },
                                        "dialogue": {
                                            "type": "ARRAY",
                                            "items": {
                                                "type": "OBJECT",
                                                "properties": {
                                                    "character": {"type": "STRING"},
                                                    "line": {"type": "STRING"},
                                                    "audio_notes": {"type": "STRING"},
                                                    "is_voiceover": {"type": "BOOLEAN"}
                                                },
                                                "required": ["character", "line"]
                                            }
                                        }
                                    },
                                    "required": ["shot_number", "start_frame", "progression", "description"]
                                }
                            }
                        },
                        "required": ["scene_number", "shots"]
                    }
                }
            },
            "required": ["title", "scenes"]
        },
        "production_notes": {
            "type": "OBJECT",
            "properties": {
                "consistency_guide": {"type": "STRING"},
                "style_guide": {"type": "STRING"},
                "key_themes": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                },
                "tone": {"type": "STRING"}
            }
        },
        "audio_design": {
            "type": "OBJECT",
            "properties": {
                "music_direction": {"type": "STRING"},
                "instrumentation": {"type": "STRING"},
                "notes": {"type": "STRING"}
            }
        }
    },
    "required": ["script_details"],
    "propertyOrdering": ["characters", "script_details", "production_notes", "audio_design"]
}

GPT_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "attributes": {"type": "string"},
                    "role": {"type": "string"}
                },
                "required": ["name", "attributes", "role"],
                "additionalProperties": False
            }
        },
        "script_details": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "duration": {"type": "string"},
                "video_summary": {"type": "string"},
                "creative_vision": {"type": "string"},
                "aspect_ratio": {"type": "string"},
                "scenes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_number": {"type": "integer"},
                            "scene_summary": {"type": "string"},
                            "setting": {"type": "string"},
                            "duration": {"type": "string"},
                            "characters": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "consistency_notes": {"type": "string"},
                            "shots": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "shot_number": {"type": "integer"},
                                        "shot_type": {"type": "string"},
                                        "duration": {"type": "string"},
                                        "subject": {"type": "string"},
                                        "description": {"type": "string"},
                                        "shot_purpose": {"type": "string"},
                                        "start_frame": {"type": "string"},
                                        "end_frame": {"type": "string"},
                                        "progression": {"type": "string"},
                                        "visual_reference": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "shot_id": {"type": "string"},
                                                    "description": {"type": "string"}
                                                },
                                                "required": ["shot_id", "description"],
                                                "additionalProperties": False
                                            }
                                        },
                                        "continuity_notes": {
                                            "type": "object",
                                            "properties": {
                                                "from_previous": {"type": "string"},
                                                "to_next": {"type": "string"},
                                                "transition_suggestion": {"type": "string"},
                                                "editing_intent": {"type": "string"}
                                            },
                                            "required": ["from_previous", "to_next", "transition_suggestion", "editing_intent"],
                                            "additionalProperties": False
                                        },
                                        "key_visual_elements": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "dialogue": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "character": {"type": "string"},
                                                    "line": {"type": "string"},
                                                    "audio_notes": {"type": "string"},
                                                    "is_voiceover": {"type": "boolean"}
                                                },
                                                "required": ["character", "line", "audio_notes", "is_voiceover"],
                                                "additionalProperties": False
                                            }
                                        }
                                    },
                                    "required": ["shot_number", "start_frame", "progression", "description", "shot_type", "duration", "subject", "shot_purpose", "end_frame", "visual_reference", "continuity_notes", "key_visual_elements", "dialogue"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["scene_number", "shots", "scene_summary", "setting", "duration", "characters", "consistency_notes"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["title", "scenes", "duration", "video_summary", "creative_vision", "aspect_ratio"],
            "additionalProperties": False
        },
        "production_notes": {
            "type": "object",
            "properties": {
                "consistency_guide": {"type": "string"},
                "style_guide": {"type": "string"},
                "key_themes": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "tone": {"type": "string"}
            },
            "required": ["consistency_guide", "style_guide", "key_themes", "tone"],
            "additionalProperties": False
        },
        "audio_design": {
            "type": "object",
            "properties": {
                "music_direction": {"type": "string"},
                "instrumentation": {"type": "string"},
                "notes": {"type": "string"}
            },
            "required": ["music_direction", "instrumentation", "notes"],
            "additionalProperties": False
        }
    },
    "required": ["script_details", "characters", "production_notes", "audio_design"],
    "additionalProperties": False
}
