"""
Schema definitions for orchestrator_agent.
Supports both Gemini and GPT structured output.

Key difference: Gemini uses additionalProperties for flexible dict,
GPT uses array of objects for agent_instructions.
"""

GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "analysis": {"type": "STRING"},
        "plan": {"type": "STRING"},
        "final_answer": {"type": "STRING"},
        "selected_agents": {
            "type": "ARRAY",
            "items": {"type": "STRING"}
        },
        "agent_instructions": {
            "type": "OBJECT",
            "additionalProperties": {"type": "STRING"}
        }
    },
    "required": ["analysis", "plan"]
}

GPT_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "plan": {"type": "string"},
        "final_answer": {"type": "string"},
        "selected_agents": {
            "type": "array",
            "items": {"type": "string"}
        },
        "agent_instructions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string"},
                    "instruction": {"type": "string"}
                },
                "required": ["agent_name", "instruction"],
                "additionalProperties": False
            }
        }
    },
    "required": ["analysis", "plan", "final_answer", "selected_agents", "agent_instructions"],
    "additionalProperties": False
}
