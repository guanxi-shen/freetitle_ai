"""
Schema definitions for answer_parser_agent.
Supports both Gemini and GPT structured output.
"""

GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "task_completion": {"type": "STRING"},
        "final_answer": {"type": "STRING"},
        "retry_rationale": {"type": "STRING"},
        "additional_assistance_needed": {"type": "STRING"},
        "route_back": {"type": "BOOLEAN"}
    },
    "required": ["task_completion"]
}

GPT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_completion": {"type": "string"},
        "final_answer": {"type": "string"},
        "retry_rationale": {"type": "string"},
        "additional_assistance_needed": {"type": "string"},
        "route_back": {"type": "boolean"}
    },
    "required": ["task_completion", "final_answer", "retry_rationale", "additional_assistance_needed", "route_back"],
    "additionalProperties": False
}
