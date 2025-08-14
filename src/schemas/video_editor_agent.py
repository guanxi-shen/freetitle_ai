"""
Schema definitions for video_editor_agent.
Supports both Gemini and GPT structured output.

Key difference: Gemini uses nullable:true, GPT uses anyOf unions with null.
"""

GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "edit_name": {"type": "STRING"},
        "selected_videos": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "filename": {"type": "STRING"},
                    "trim": {
                        "type": "OBJECT",
                        "properties": {
                            "start": {"type": "NUMBER"},
                            "end": {"type": "NUMBER"}
                        },
                        "required": ["start", "end"],
                        "nullable": True
                    },
                    "mute_audio": {
                        "type": "BOOLEAN",
                        "nullable": True
                    }
                },
                "required": ["filename", "trim", "mute_audio"]
            }
        },
        "transitions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "from": {"type": "STRING", "nullable": True},
                    "to": {"type": "STRING", "nullable": True},
                    "type": {"type": "STRING"},
                    "duration": {"type": "NUMBER"}
                },
                "required": ["from", "to", "type", "duration"]
            }
        },
        "aspect_ratio": {"type": "STRING"},
        "add_audio": {"type": "BOOLEAN"},
        "selected_audio": {"type": "STRING", "nullable": True},
        "audio_volume": {"type": "NUMBER"},
        "mix_original_audio": {"type": "BOOLEAN"},
        "notes": {"type": "STRING"}
    },
    "required": ["edit_name", "selected_videos", "transitions", "aspect_ratio", "add_audio", "selected_audio", "audio_volume", "mix_original_audio", "notes"]
}

GPT_SCHEMA = {
    "type": "object",
    "properties": {
        "edit_name": {"type": "string"},
        "selected_videos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "trim": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "number"},
                                    "end": {"type": "number"}
                                },
                                "required": ["start", "end"],
                                "additionalProperties": False
                            },
                            {"type": "null"}
                        ]
                    },
                    "mute_audio": {
                        "anyOf": [
                            {"type": "boolean"},
                            {"type": "null"}
                        ]
                    }
                },
                "required": ["filename", "trim", "mute_audio"],
                "additionalProperties": False
            }
        },
        "transitions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"}
                        ]
                    },
                    "to": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"}
                        ]
                    },
                    "type": {"type": "string"},
                    "duration": {"type": "number"}
                },
                "required": ["from", "to", "type", "duration"],
                "additionalProperties": False
            }
        },
        "aspect_ratio": {"type": "string"},
        "add_audio": {"type": "boolean"},
        "selected_audio": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"}
            ]
        },
        "audio_volume": {"type": "number"},
        "mix_original_audio": {"type": "boolean"},
        "notes": {"type": "string"}
    },
    "required": ["edit_name", "selected_videos", "transitions", "aspect_ratio", "add_audio", "selected_audio", "audio_volume", "mix_original_audio", "notes"],
    "additionalProperties": False
}
