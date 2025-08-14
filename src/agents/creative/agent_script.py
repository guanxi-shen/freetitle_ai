"""
Creative production agents for AI Video Studio
"""

import json
import time
from datetime import datetime
from ...core.state import RAGState
from ...core.llm import get_llm
from ...prompts.script import get_script_agent_prompt
from ..base import get_agent_context, emit_event, clean_json_response


def get_script_llm_choice(state: RAGState) -> str:
    """Get LLM model for script agent. Always returns gemini."""
    return "gemini"


def script_agent(state: RAGState, task_instruction: str = None) -> RAGState:
    """Script writing agent - creates video scripts based on user requirements and creative direction"""
    print("[Script Agent] Starting script generation...")
    start_time = time.time()
    
    # Set current agent for event tracking
    state["current_agent"] = "script_agent"
    
    # Emit agent started event for streaming
    state = emit_event(state, "agent_started", {"agent": "script_agent"}, agent_name="script_agent")
    
    # Note: system_instruction will be set after loading prompt template

    user_query = state["user_query"]
    instruction = task_instruction or state["agent_instructions"].get("script_agent", "Create a compelling video script")
    
    if "component_timings" not in state:
        state["component_timings"] = {}
    
    # Get template config with dynamic sections
    prompt_config = get_script_agent_prompt(state)

    # Build agent-specific context with dynamic sections
    agent_specific_parts = []
    if prompt_config.get("document_instruction"):
        agent_specific_parts.append(prompt_config["document_instruction"])
    if prompt_config.get("planning_guidance"):
        agent_specific_parts.append(prompt_config["planning_guidance"])
    if prompt_config.get("cinematic_instruction"):
        agent_specific_parts.append(prompt_config["cinematic_instruction"])

    agent_specific = "\n\n".join(agent_specific_parts) if agent_specific_parts else None

    # Build complete context with all dynamic content
    from ..base import build_full_context
    context_content = build_full_context(
        state,
        agent_name="script_agent",
        user_query=user_query,
        instruction=instruction,
        agent_specific_context=agent_specific,
        context_summary=True,
        window_memory=True,
        include_generated_content=True,
        include_reference_images=True,
        include_image_annotations=True,
        include_generated_assets=True
    )

    # Get 100% static template
    script_prompt = prompt_config["template"]

    print(f"[Script Agent] JSON mode: ENABLED (response_schema='script_agent')")

    # Get or create cache
    from src.core.cache_registry import get_or_create_cache
    from ...core.config import SCRIPT_AGENT_USE_RAW_DOCUMENTS

    has_docs = bool(state.get("enterprise_resources", {}).get("documents", []))
    cache_docs = has_docs and SCRIPT_AGENT_USE_RAW_DOCUMENTS

    cached_content = get_or_create_cache(
        agent_name="script",
        system_instruction=script_prompt,
        state=state,
        prompt_variation="gemini" if cache_docs else None,
        include_documents=cache_docs
    )

    # Initialize LLM with prompt template as system_instruction
    script_llm = get_llm(
        model="gemini",
        gemini_configs={'max_output_tokens': 14500, 'temperature': 1.0},
        system_instruction=script_prompt,
        cached_content_name=cached_content
    )

    # Load enterprise documents as content parts if available and flag is enabled
    document_parts = []
    has_documents = prompt_config.get("has_documents", False)

    if has_documents:
        from ..base import load_enterprise_documents
        document_parts = load_enterprise_documents(state, agent_name="Script Agent")

    # Context and prompt are already available in the function scope if needed for debugging

    try:
        # Emit processing event
        state = emit_event(state, "processing", {"message": "Generating script..."}, agent_name="script_agent")

        response = script_llm.invoke(
            context_content,  # Only dynamic context, prompt is in system_instruction
            add_context=False,  # Don't concatenate, prompt already in system_instruction
            response_schema="script_agent",
            documents=document_parts if document_parts else None,
            state=state,
            stream_callback=lambda event_type, content: emit_event(
                state,
                f"llm_{event_type}",
                {"content": content},
                agent_name="script_agent"
            ) if content else None
        )
        
        
        # Parse LLM response
        if isinstance(response, str):
            parsed_response = json.loads(response)
        else:
            parsed_response = response
        
        
        response_text = parsed_response['content'][1]['text']
        thinking = parsed_response['content'][0]['thinking']
        
        
        # Clean JSON response from markdown code blocks
        response_text = clean_json_response(response_text)
        
        # Parse the JSON script output
        script_data = json.loads(response_text)
        print(f"[Script Agent] ✓ Script JSON parsed successfully")
        print(f"[Script Agent] Generated script: {script_data.get('script_details', {}).get('title', 'Untitled')}")
        print(f"[Script Agent] Scenes: {len(script_data.get('script_details', {}).get('scenes', []))}")
        
        # Emit script generated event with key data
        state = emit_event(state, "script_generated", {
            "title": script_data.get('script_details', {}).get('title', 'Untitled'),
            "scenes": len(script_data.get('script_details', {}).get('scenes', [])),
            "duration": script_data.get('script_details', {}).get('duration', 'Unknown')
        }, agent_name="script_agent")
        
        # Store script output in state (backward compatibility)
        state["script_output"] = {
            "script_data": script_data,
            "thinking": thinking,
            "timestamp": datetime.now().isoformat(),
            "agent": "script_agent"
        }
        
        # Store in generated_scripts for persistence (single source of truth - just the script data)
        state["generated_scripts"] = script_data
        print(f"[Script Agent] Stored script in generated_scripts: {script_data.get('script_details', {}).get('title', 'Untitled')}")
        
        # Store thinking process for UI display
        if "thinking_processes" not in state:
            state["thinking_processes"] = {}
        state["thinking_processes"]["script_agent"] = thinking
        
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        # Handle parsing errors gracefully
        error_msg = f"Error parsing script response: {str(e)}"
        print(f"[Script Agent] Error: {error_msg}")
        
        # Emit essential error info for debugging if needed
        import traceback
        state = emit_event(state, "error", {
            "agent": "script_agent",
            "error": error_msg,
            "traceback": traceback.format_exc()
        }, agent_name="script_agent")
        
        state["script_output"] = {
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
            "agent": "script_agent"
        }
        
        if "thinking_processes" not in state:
            state["thinking_processes"] = {}
        state["thinking_processes"]["script_agent"] = f"Script generation encountered an error: {error_msg}"
    
    # Record timing
    execution_time = time.time() - start_time
    state["component_timings"]["script_agent"] = execution_time
    print(f"[Script Agent] Completed in {execution_time:.2f}s")

    # Emit agent ended event
    state = emit_event(state, "agent_ended", {"agent": "script_agent"}, agent_name="script_agent")

    return state