"""System agents - core video production coordination and workflow management"""

import json
import time
import uuid
from datetime import datetime
from ...core.state import RAGState
from ...core.llm import get_llm
from ...core.config import ROLLING_WINDOW_SIZE
from ..base import emit_event
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

def memory_updater_agent(state: RAGState) -> RAGState:
    """Memory updater with rolling window logic and context summarization"""
    print("[Memory Updater Agent] Starting memory update...")
    start_time = time.time()
    
    # Set current agent for event tracking
    state["current_agent"] = "memory_updater"

    # Import emit_event for agent tracking
    from ..base import emit_event

    # Emit agent started event for frontend tracking
    state = emit_event(state, "agent_started", {"agent": "memory_updater"}, agent_name="memory_updater")

    memory_update_llm = get_llm(
        fallback_configs={'max_tokens': 3000, 'temperature': 1.0},
        system_instruction="You are a conversation summarization specialist. Create concise, informative summaries."
    )
    
    if "component_timings" not in state:
        state["component_timings"] = {}

    # Defensive initialization
    if "metadata" not in state:
        state["metadata"] = {}
    if "conversation_history" not in state:
        state["conversation_history"] = []
    if "messages" not in state:
        state["messages"] = []

    # Create conversation record
    conversation_record = {
        "turn_number": state["turn_number"],
        "timestamp": state["timestamp"],
        "query": state["user_query"],
        "answer": state.get("final_answer", "No answer generated"),
        "thinking_processes": state.get("thinking_processes", {}),
        "component_timings": state.get("component_timings", {}),
        "total_execution_time": state.get("total_execution_time", 0)
    }

    state["conversation_history"].append(conversation_record)
    
    # Add AI message as dict for JSON serialization
    state["messages"].append({"role": "assistant", "content": state.get("final_answer", "No answer generated")})
    
    if len(state["conversation_history"]) > ROLLING_WINDOW_SIZE:
        if len(state["conversation_history"]) == ROLLING_WINDOW_SIZE + 1:
            existing_summary = state.get("context_summary", "")
            conversations_to_summarize = state["conversation_history"][:-ROLLING_WINDOW_SIZE]
            summary_content = ""
            for conv in conversations_to_summarize:
                summary_content += f"USER: {conv['query']}\nSYSTEM: {conv['answer']}\n\n"

            # NEW: Get ALL expert conversations up to current global turn threshold
            expert_conversations = state.get("expert_conversation_history", {})
            global_turn_threshold = state["turn_number"] - ROLLING_WINDOW_SIZE

            # Summarize ALL expert messages with global_turn <= threshold
            for expert_id, conversations in expert_conversations.items():
                for conv in conversations:
                    # Use global turn to determine what to summarize
                    if conv.get('global_turn', 0) <= global_turn_threshold:
                        summary_content += f"\n[Expert {expert_id}] Turn {conv.get('global_turn', 0)}:\n"
                        summary_content += f"USER: {conv.get('user_query', '')}\n"
                        summary_content += f"EXPERT: {conv.get('expert_response', '')}\n\n"

            # Build context content with dynamic data
            if existing_summary:
                context_content = f"""### Existing Summary:
{existing_summary}

### New Conversations to Include:
{summary_content}"""
                template = MEMORY_UPDATE_SUMMARY_PROMPT_TEMPLATE["template"]
            else:
                context_content = f"""### Conversations to Summarize:
{summary_content}"""
                template = MEMORY_SUMMARY_PROMPT_TEMPLATE["template"]

            try:
                response = memory_update_llm.invoke(
                    template,
                    add_context=True,
                    context_content=context_content,
                    state=state,
                    stream_callback=lambda event_type, content: emit_event(
                        state,
                        f"llm_{event_type}",
                        {"content": content},
                        agent_name="memory_updater"
                    ) if content else None
                )
                
                # Handle both JSON and plain text responses
                if isinstance(response, str):
                    if response.startswith('{'):
                        # JSON response
                        parsed_response = json.loads(response)
                        new_summary = parsed_response['content'][1]['text']
                        thinking = parsed_response['content'][0]['thinking']
                    else:
                        # Plain text response - use directly
                        new_summary = response
                        thinking = "Direct text response"
                else:
                    # Empty or invalid response
                    raise ValueError(f"Empty or invalid response from LLM. Response type: {type(response)}, Content: {response}")
                
                state["context_summary"] = new_summary
                summary_action = f"Generated comprehensive summary (existing: {'Yes' if existing_summary else 'No'}) from {len(conversations_to_summarize)} conversations. {thinking}"
                
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                import traceback
                full_error = traceback.format_exc()
                summary_action = f"Error generating context summary. Error: {str(e)}. Response was: {response[:500] if 'response' in locals() else 'No response'}. Full trace: {full_error}"
        
        elif len(state["conversation_history"]) > ROLLING_WINDOW_SIZE + 1:
            rolled_out_conv = state["conversation_history"][-(ROLLING_WINDOW_SIZE + 1)]
            current_summary = state.get("context_summary", "")

            # Build context content with dynamic data
            context_content = f"""### Current Summary:
{current_summary}

### New Conversation to Include:
USER: {rolled_out_conv['query']}
SYSTEM: {rolled_out_conv['answer']}"""

            template = MEMORY_UPDATE_CONVERSATION_PROMPT_TEMPLATE["template"]

            try:
                response = memory_update_llm.invoke(
                    template,
                    add_context=True,
                    context_content=context_content,
                    state=state,
                    stream_callback=lambda event_type, content: emit_event(
                        state,
                        f"llm_{event_type}",
                        {"content": content},
                        agent_name="memory_updater"
                    ) if content else None
                )
                
                # Handle both JSON and plain text responses
                if isinstance(response, str):
                    if response.startswith('{'):
                        # JSON response
                        parsed_response = json.loads(response)
                        updated_summary = parsed_response['content'][1]['text']
                        thinking = parsed_response['content'][0]['thinking']
                    else:
                        # Plain text response - use directly
                        updated_summary = response
                        thinking = "Direct text response"
                else:
                    # Empty or invalid response
                    raise ValueError(f"Empty or invalid response from LLM. Response type: {type(response)}, Content: {response}")
                
                state["context_summary"] = updated_summary
                summary_action = f"Updated context summary with rolled-out conversation. {thinking}"
                
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                import traceback
                full_error = traceback.format_exc()
                summary_action = f"Failed to update context summary. Error: {str(e)}. Response was: {response[:500] if 'response' in locals() else 'No response'}. Full trace: {full_error}"
        else:
            summary_action = "No summary update needed"
        
        state["conversation_history"] = state["conversation_history"][-ROLLING_WINDOW_SIZE:]
    else:
        summary_action = f"Window not full yet ({len(state['conversation_history'])}/{ROLLING_WINDOW_SIZE})"
    
    print(f"[Memory Updater Agent] Memory action: {summary_action}")

    state["window_memory"] = state["conversation_history"]

    # Enterprise document analysis moved to memory_manager (runs before orchestrator)
    # This ensures orchestrator has business context when answering

    # Window expert conversations by global turn
    expert_conversations = state.get("expert_conversation_history", {})
    expert_window = {}
    current_global_turn = state.get("turn_number", 0)
    window_start_turn = max(1, current_global_turn - ROLLING_WINDOW_SIZE + 1)

    for expert_id, conversations in expert_conversations.items():
        # Only keep expert messages within the global turn window
        windowed_convs = [
            conv for conv in conversations
            if conv.get('global_turn', 0) >= window_start_turn
        ]
        if windowed_convs:
            expert_window[expert_id] = windowed_convs

    state["expert_window_memory"] = expert_window

    # Count expert messages in window for logging
    expert_msg_count = sum(len(convs) for convs in expert_window.values())

    state["metadata"]["memory_updated"] = True
    state["thinking_processes"]["memory_updater"] = (
        f"Updated rolling window memory. Main window: {len(state['window_memory'])}/{ROLLING_WINDOW_SIZE}. "
        f"Expert conversations in window: {expert_msg_count} messages across {len(expert_window)} experts. "
        f"Summary action: {summary_action}"
    )
    
    execution_time = time.time() - start_time
    state["component_timings"]["memory_updater"] = execution_time
    print(f"[Memory Updater Agent] Completed in {execution_time:.2f}s")
    
    return state