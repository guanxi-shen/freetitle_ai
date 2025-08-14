"""Streaming wrapper for LangGraph workflow execution"""

from typing import AsyncIterator, Dict, Any, Callable, Optional, List
import json
import asyncio
import os
from datetime import datetime

from .workflow import build_rag_workflow
from .state import RAGState
from ..api.redis_state import RedisStateManager

# Expert and main workflow states are now completely isolated
# No need for yielding as there's no contention

async def stream_workflow(
    query: str,
    session_id: str,
    stream_callback: Callable,
    reference_images: Optional[List[str]] = None,
    tool_selections: Optional[Dict[str, List[str]]] = None,
    expert_selections: Optional[List[str]] = None
) -> None:
    """
    Stream workflow execution with real-time updates.

    Reference: https://langchain-ai.github.io/langgraph/how-tos/streaming/#stream-from-an-agent

    Args:
        query: User query to process
        session_id: Session identifier
        stream_callback: Callback function for streaming events
        reference_images: Optional list of reference image paths
        tool_selections: Optional dict of tool selections per agent
        expert_selections: Optional list of active expert agents

    Yields:
        Stream events as they occur
    """
    
    # Initialize Redis state manager
    # Use REDIS_URL_TEST for now (production will use REDIS_URL later)
    redis_url = os.getenv("REDIS_URL_TEST")
    if not redis_url:
        raise ValueError("[Streaming] REDIS_URL_TEST must be set")
    
    state_manager = RedisStateManager(redis_url)
    
    # Build workflow graph
    graph = build_rag_workflow()
    
    # Try to load existing state from Redis first
    existing_state = await state_manager.get_state(session_id)

    if existing_state:
        # Resume from existing state (preserves uploaded documents even if no messages yet)
        message_count = len(existing_state.get('messages', []))
        print(f"[Streaming] Resuming session {session_id} with {message_count} previous message(s)")
        print(f"[Streaming] DEBUG: State keys from Redis = {list(existing_state.keys())}")
        print(f"[Streaming] DEBUG: Has enterprise_resources = {bool(existing_state.get('enterprise_resources'))}")
        initial_state = existing_state
        # Update with new query
        initial_state["user_query"] = query
        # Increment turn number
        initial_state["turn_number"] = existing_state.get("turn_number", 0) + 1
        # Ensure session_id and thread_id are present (may be missing in older states)
        initial_state["session_id"] = session_id
        initial_state["thread_id"] = session_id
        # Clear previous turn's final answer to prevent duplicate events
        initial_state["final_answer"] = None
        # Add new reference images if provided
        if reference_images:
            initial_state["reference_images"] = reference_images
        # Update tool selections in user_preferences
        if tool_selections:
            if "user_preferences" not in initial_state:
                initial_state["user_preferences"] = {}
            initial_state["user_preferences"]["tool_selections"] = tool_selections
        # Set default tool selections if not provided
        elif "user_preferences" not in initial_state or "tool_selections" not in initial_state.get("user_preferences", {}):
            if "user_preferences" not in initial_state:
                initial_state["user_preferences"] = {}
            initial_state["user_preferences"]["tool_selections"] = {
                "video_agent": ["google_veo_i2v"]
            }
        # Update expert selections
        if expert_selections:
            initial_state["expert_selections"] = expert_selections
        elif "expert_selections" not in initial_state:
            initial_state["expert_selections"] = []
    else:
        # Create new initial state
        # NOTE: Do NOT initialize enterprise_resources or enterprise_agent_output here
        # They are created by upload endpoint and preserved when state exists
        print(f"[Streaming] Starting new session {session_id}")
        initial_state = {
            "user_query": query,
            "session_id": session_id,
            "thread_id": session_id,
            "messages": [],
            "turn_number": 1,
            "reference_images": reference_images or [],
            "user_preferences": {
                "tool_selections": tool_selections or {
                    "video_agent": ["google_veo_i2v"]
                }
            },
            "expert_selections": expert_selections or [],
            "asset_urls": {
                "user_references": [],
                "storyboard_frames": [],
                "supplementary_assets": [],
                "generated_videos": [],
                "edited_videos": [],
                "generated_audio": []
            },
            "audio_generation_metadata": {},
            "video_generation_metadata": {}
        }
    
    # Stream with multiple modes - returns tuples of (mode, data)
    # Track what we've already sent to avoid duplicates
    last_final_answer = None
    last_message_count = 0

    async for mode, data in graph.astream(
        initial_state,
        config={"configurable": {"thread_id": session_id}},
        stream_mode=["updates", "custom", "values"]  # Changed "messages" to "custom" for StreamWriter support
    ):
        # Process each stream chunk with deduplication
        await process_stream_data(
            mode, data, initial_state, stream_callback, state_manager, session_id,
            last_final_answer, last_message_count
        )

        # Update tracking for deduplication
        if mode == "values" and isinstance(data, dict):
            if "final_answer" in data:
                last_final_answer = data["final_answer"]
            if "messages" in data:
                last_message_count = len(data["messages"])

    # Clean up
    await state_manager.close()

async def process_stream_data(mode, data, initial_state, stream_callback, state_manager, session_id,
                             last_final_answer=None, last_message_count=0):
    """Process a single stream data chunk with deduplication"""
    if mode == "updates":
        # Get the actual node name from the update
        # LangGraph sends updates as {node_name: node_data}
        if isinstance(data, dict) and len(data) == 1:
            current_node = list(data.keys())[0]
            node_data = data[current_node]
            current_agent = node_data.get("current_agent", current_node) if isinstance(node_data, dict) else current_node
        else:
            current_node = data.get("current_node", "unknown") if isinstance(data, dict) else "unknown"
            current_agent = data.get("current_agent") if isinstance(data, dict) else None
        
        # Only send node transition if we have actual node info
        if current_node and current_node != "unknown":
            await stream_callback({
                "type": "node_transition",
                "node": current_node,
                "agent": current_agent,
                "timestamp": datetime.now().isoformat()
            })
        
        # Send agent thinking if available
        thinking = data.get("thinking_processes", {})
        if thinking and current_agent:
            agent_thinking = thinking.get(current_agent)
            if agent_thinking:
                await stream_callback({
                    "type": "agent_thinking",
                    "agent": current_agent,
                    "content": agent_thinking
                })
        
        # Send execution plan if available
        if "execution_plan" in data and data["execution_plan"]:
            await stream_callback({
                "type": "execution_plan",
                "content": data["execution_plan"]
            })
        
        # Send selected agents if available  
        if "selected_agents" in data and data["selected_agents"]:
            await stream_callback({
                "type": "agents_selected",
                "agents": data["selected_agents"]
            })
        
        # No longer checking for streaming_events - using StreamWriter instead
                
    elif mode == "custom":
        # Direct streaming from StreamWriter - forward immediately
        if data is not None:
            # Filter out message_added events - they're redundant with token streaming
            if isinstance(data, dict) and data.get("type") == "message_added":
                return  # Skip message_added events
            await stream_callback(data)
                    
    elif mode == "values":
        # Full state snapshot - save to Redis
        # Expert states are now completely isolated, no preservation needed
        # OPTIMIZATION: Commented out for testing - UI doesn't need intermediate saves
        # Only save at workflow end (memory_updater) for session persistence
        current_agent = data.get("current_agent", "")
        if current_agent == "memory_updater":
            await state_manager.save_state(session_id, data)
        # else:
        #     # Skipping intermediate save for agent: {current_agent}
        #     pass
        
        # Send current agent and thinking process
        if "current_agent" in data and data["current_agent"]:
            await stream_callback({
                "type": "agent_active",
                "agent": data["current_agent"],
                "thinking": data.get("thinking_processes", {}).get(data["current_agent"], "")
            })
        
        # Skip sending message_added events - they're redundant with token streaming
        # Messages are still tracked in state for conversation history
        # if "messages" in data and len(data["messages"]) > last_message_count:
        #     # Commented out - we already stream tokens in real-time
        #     pass

        # State-based emission REMOVED - agents now emit events directly via StreamWriter
        # This prevents duplicate events when state persists across turns
        # Events are emitted in real-time by:
        # - script_agent.py:114 (script_generated)
        # - character_agent.py:539 (characters_generated)
        # - storyboard_agent.py:594 (storyboard_generated)
        # - storyboard_agent.py:309 (storyboard_frame_generated - per frame)
        # - audio_agent.py:267 (audio_generated)
        # - audio_agent.py:105 (audio_track_generated - per track)
        # - video_task_monitor.py:268 (videos_generated)

        # Send final answer if available (only if it's new)
        if "final_answer" in data and data["final_answer"] and data["final_answer"] != last_final_answer:
            await stream_callback({
                "type": "final_answer",
                "content": data["final_answer"]
            })