"""System agents - core video production coordination and workflow management"""

import json
import time
import uuid
from datetime import datetime
from ...core.state import RAGState
from ...core.llm import get_llm
from ...core.config import ROLLING_WINDOW_SIZE
from ...prompts import (
    ORCHESTRATOR_PROMPT_TEMPLATE,
    ORCHESTRATOR_RETRY_PROMPT_TEMPLATE,
    ANSWER_SYNTHESIS_BASE_PROMPT_TEMPLATE,
    ANSWER_SYNTHESIS_WITH_SUFFICIENCY_TEMPLATE,
    MEMORY_SUMMARY_PROMPT_TEMPLATE,
    MEMORY_UPDATE_SUMMARY_PROMPT_TEMPLATE,
    MEMORY_UPDATE_CONVERSATION_PROMPT_TEMPLATE,
    AGENT_REGISTRY
)
from ..base import emit_event, clean_json_response

def get_orchestrator_llm_choice(state: RAGState) -> str:
    """Get LLM model for orchestrator agent. Always returns gemini."""
    return "gemini"

def orchestrator_agent(state: RAGState) -> RAGState:
    """Orchestrator agent that acts as Creative Assistant - analyzes video requests, creates production plan, and selects creative agents with instructions. Handles retry scenarios."""
    print("[Orchestrator Agent] Starting creative orchestration...")
    start_time = time.time()
    
    # Set current agent for event tracking
    state["current_agent"] = "orchestrator"

    # Import utilities from parent module
    from .. import base as agents

    # Emit agent started event for frontend tracking
    state = agents.emit_event(state, "agent_started", {"agent": "orchestrator"}, agent_name="orchestrator")

    # Note: system_instruction will be set based on retry vs standard path
    
    user_query = state["user_query"]
    context_summary = state.get("context_summary", "")
    window_memory = state.get("window_memory", [])
    
    if "component_timings" not in state:
        state["component_timings"] = {}
    
    available_agents = "\n".join([agents.get_agent_info(name) for name in AGENT_REGISTRY.keys()])
    
    # Check if this is a retry scenario - based on retry context being set
    retry_context = state.get("orchestrator_retry_context", {})
    is_retry = retry_context.get("is_retry", False)
    
    if is_retry:
        print(f"[Orchestrator Agent] Retry attempt {state.get('retry_count', 0) + 1}")
    else:
        print(f"[Orchestrator Agent] Processing query: {user_query[:100]}...")
    
    if is_retry:
        # Handle retry scenario with specific prompt
        # Note: retry_count was already incremented in answer_parser when routing back
        
        # Format previous attempts summary
        previous_attempts = retry_context.get("previous_attempts", [])
        attempts_summary = ""
        for attempt in previous_attempts:
            attempts_summary += f"Attempt {attempt['attempt_number']}: Used {attempt.get('agents_used', [])} -> {attempt.get('outcome', 'unknown')}. "
        
        # Format additional assistance needed
        additional_assistance_needed = retry_context.get("additional_assistance_needed", "No specific assistance specified")
        
        # Web search protocol disabled - commented out
        current_attempt = state["retry_count"]
        # if current_attempt >= 2:
        #     web_search_protocol = """
        #
        # WEB SEARCH PROTOCOL (3rd attempt):
        # - Web search is disabled in current implementation
        # - Focus on generating creative content directly"""

        # Build retry-specific context
        retry_specific = f"""### Retry Information:
Retry Reason: {retry_context.get("retry_reason", "Unknown")}
Current Attempt: {current_attempt}
Previous Attempts Summary:
{attempts_summary}
Additional Assistance Needed: {additional_assistance_needed}

### Available Agents:
{available_agents}"""

        # Get static template (no variables)
        orchestrator_prompt = ORCHESTRATOR_RETRY_PROMPT_TEMPLATE["template"]
        orchestrator_schema = ORCHESTRATOR_RETRY_PROMPT_TEMPLATE["schema"]

        # Clear retry context after use
        state["orchestrator_retry_context"] = {}
    else:
        # Standard orchestration
        if "retry_count" not in state:
            state["retry_count"] = 0
        if "previous_attempts" not in state:
            state["previous_attempts"] = []

        # Build standard agent list context
        retry_specific = f"""### Available Agents:
{available_agents}"""

        # Get static template (no variables)
        orchestrator_prompt = ORCHESTRATOR_PROMPT_TEMPLATE["template"]
        orchestrator_schema = ORCHESTRATOR_PROMPT_TEMPLATE["schema"]

    # Get or create cache (static system instruction only)
    # TODO: Test enterprise/reference caching later - keeping clean separation for now
    from src.core.cache_registry import get_or_create_cache
    prompt_variation = "retry" if is_retry else "standard"
    cached_content = get_or_create_cache(
        agent_name="orchestrator",
        system_instruction=orchestrator_prompt,
        state=state,
        prompt_variation=prompt_variation
        # include_enterprise=True,  # Commented out - keep enterprise in dynamic context for now
        # include_references=True   # Commented out - keep references in dynamic context for now
    )

    # Initialize LLM
    orchestrator_llm = get_llm(
        model="gemini",
        gemini_configs={'max_output_tokens': 6000, 'temperature': 1.0, 'top_p': 0.95},
        system_instruction=orchestrator_prompt,
        cached_content_name=cached_content
    )

    # Build complete context with all dynamic content
    context_content = agents.build_full_context(
        state,
        agent_name="orchestrator",
        user_query=user_query,
        agent_specific_context=retry_specific,
        context_summary=True,
        window_memory=True,
        include_generated_content=True,
        include_reference_images=True,
        include_image_annotations=True,
        include_tool_selections=True,
        include_expert_info=True
    )
    
    
    # Context and prompt are already available in the function scope if needed for debugging
    
    # JSON validation with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"[Orchestrator] Attempt {attempt + 1}/{max_retries} - Invoking with schema: orchestrator_agent")

            response = orchestrator_llm.invoke(
                context_content,  # Only dynamic context, prompt is in system_instruction
                add_context=False,  # Don't concatenate, prompt already in system_instruction
                response_schema="orchestrator_agent",  # Auto-selects Gemini or GPT schema
                state=state,
                stream_callback=lambda event_type, content: emit_event(
                    state,
                    f"llm_{event_type}",
                    {"content": content},
                    agent_name="orchestrator"
                ) if content else None
            )

            parsed_response = json.loads(response)
            response_text = parsed_response['content'][1]['text']
            thinking = parsed_response['content'][0]['thinking']
            orchestrator_data = json.loads(clean_json_response(response_text))
            print(f"[Orchestrator] Parsed JSON - keys: {list(orchestrator_data.keys())}")

            # Defensive conversion: Handle both array and dict formats for agent_instructions
            if "agent_instructions" in orchestrator_data:
                value = orchestrator_data["agent_instructions"]
                if isinstance(value, list):
                    # Array format (GPT) → Convert to dict
                    converted = {item["agent_name"]: item.get("instruction", "")
                                for item in value if isinstance(item, dict) and "agent_name" in item}
                    orchestrator_data["agent_instructions"] = converted if converted else {}
                    print(f"[Orchestrator] Converted array→dict: {len(converted)} agents")
                elif isinstance(value, dict):
                    pass  # Dict format (Gemini/fallback) → Pass through
                else:
                    print(f"[Orchestrator] WARNING: Unexpected type {type(value).__name__}, using empty dict")
                    orchestrator_data["agent_instructions"] = {}

            # Check if this is a direct answer response
            if orchestrator_data.get("plan") == "direct_answer" and "final_answer" in orchestrator_data:
                print("[Orchestrator] Direct answer path")
                state["execution_plan"] = f"Analysis: {orchestrator_data['analysis']}\nPlan: Direct answer provided"
                state["selected_agents"] = []
                state["agent_instructions"] = {}
                state["final_answer"] = orchestrator_data["final_answer"]
                state["thinking_processes"]["orchestrator"] = thinking
                break

            # Otherwise, check for agent routing format
            required_fields = ["analysis", "plan", "selected_agents", "agent_instructions"]
            missing_fields = [f for f in required_fields if f not in orchestrator_data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")

            selected_agents = orchestrator_data["selected_agents"]

            # Parse parallel execution signals (format: "parallel:character_agent,supplementary_agent")
            processed_agents = []
            for agent in selected_agents:
                if isinstance(agent, str) and agent.startswith("parallel:"):
                    # Extract exact agent names (comma-separated)
                    agent_names = agent.replace("parallel:", "").split(",")
                    agent_names = [name.strip() for name in agent_names]  # Remove whitespace
                    processed_agents.append(agent_names)
                    print(f"[Orchestrator] Parsed parallel group: {agent_names}")
                else:
                    processed_agents.append(agent)

            selected_agents = processed_agents

            # Validate agents (handle both strings and nested lists)
            def validate_agent(agent):
                if isinstance(agent, list):
                    return all(a in AGENT_REGISTRY for a in agent)
                return agent in AGENT_REGISTRY

            valid_agents = [agent for agent in selected_agents if validate_agent(agent)]
            invalid_agents = [agent for agent in selected_agents if not validate_agent(agent)]

            if invalid_agents:
                print(f"[Orchestrator] WARNING: Invalid agents filtered: {invalid_agents}")

            if valid_agents:
                print(f"[Orchestrator] Routing to agents: {valid_agents}")
                state["execution_plan"] = f"Analysis: {orchestrator_data['analysis']}\nPlan: {orchestrator_data['plan']}"
                state["selected_agents"] = valid_agents
                state["agent_instructions"] = orchestrator_data["agent_instructions"]
                state["thinking_processes"]["orchestrator"] = thinking
                break
            else:
                raise ValueError(f"No valid agents. Requested: {selected_agents}")
                
        except json.JSONDecodeError as e:
            print(f"[Orchestrator] ERROR: JSON decode - {e} at line {e.lineno}")
            print(f"[Orchestrator] Raw response (first 500 chars): {response[:500]}")
            if attempt == max_retries - 1:
                print(f"[Orchestrator] FATAL: Max retries reached")
        except KeyError as e:
            print(f"[Orchestrator] ERROR: Missing key {e}")
            if 'orchestrator_data' in locals():
                print(f"[Orchestrator] Available keys: {list(orchestrator_data.keys())}")
            if attempt == max_retries - 1:
                print(f"[Orchestrator] FATAL: Max retries reached")
        except ValueError as e:
            print(f"[Orchestrator] ERROR: Validation - {e}")
            if attempt == max_retries - 1:
                print(f"[Orchestrator] FATAL: Max retries reached")
        except Exception as e:
            print(f"[Orchestrator] ERROR: {type(e).__name__} - {e}")
            import traceback
            print(f"[Orchestrator] Traceback:\n{traceback.format_exc()}")
            if attempt == max_retries - 1:
                # Display error to user, don't fallback to any agent
                error_msg = f"I apologize, but I encountered an error understanding your request after {max_retries} attempts. Please try rephrasing your request or contact support if this persists."

                state["selected_agents"] = []  # No agents selected
                state["final_answer"] = error_msg
                state["execution_plan"] = f"Orchestrator error: {str(e)}"
                state["thinking_processes"]["orchestrator"] = f"Failed after {max_retries} attempts: {str(e)}"
                state["orchestrator_error"] = {
                    "error": str(e),
                    "raw_response": response[:500] if response else "No response",
                    "attempts": max_retries
                }
                break  # Exit retry loop
            else:
                orchestrator_prompt += f"\n\nPrevious attempt error: {str(e)}. Please ensure valid JSON format."
    
    execution_time = time.time() - start_time
    state["component_timings"]["orchestrator"] = execution_time
    print(f"[Orchestrator Agent] Completed in {execution_time:.2f}s")
    
    return state