"""Workflow builder and routing for AI Video Studio system"""

import os
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import ContextThreadPoolExecutor
from concurrent.futures import as_completed
import copy
from .state import RAGState
from ..agents.system import (
    orchestrator_agent,
    answer_parser_agent,
    memory_manager_agent,
    memory_updater_agent,
    video_task_monitor
)
from ..agents.creative import (
    script_agent,
    character_agent,
    storyboard_agent,
    supplementary_agent,
    audio_agent,
    video_agent,
    video_editor_agent
)


def should_monitor_videos(state: RAGState) -> str:
    """Route after video agent: check if video tasks need monitoring"""
    tasks = state.get("video_generation_tasks", [])

    # Check if there are submitted tasks that need monitoring
    submitted_tasks = [t for t in tasks if t.get("status") == "submitted"]

    if submitted_tasks:
        print(f"[Workflow Router] {len(submitted_tasks)} video tasks submitted (out of {len(tasks)} total), routing to monitor")
        return "video_task_monitor"
    else:
        print(f"[Workflow Router] No video tasks to monitor, routing to answer_parser")
        return "answer_parser"


# Supplementary monitor removed - now using synchronous blocking pattern


def retry_router(state: RAGState) -> str:
    """Route after answer parser: check if task completion retry is needed"""
    from .config import DEFAULT_MAX_RETRIES
    
    retry_count = state.get("retry_count", 0)
    # Use state-specific max_retries if provided, otherwise use default
    max_retries = state.get("max_retries", DEFAULT_MAX_RETRIES)
    
    # Check if task is incomplete and we haven't exceeded max retries
    task_status = state.get("task_completion", {})
    print(f"[Workflow Router] Retry check: attempt {retry_count}/{max_retries}, status: {task_status.get('status', 'unknown')}")
    
    if (retry_count < max_retries and 
        task_status.get("status") == "incomplete"):
        print(f"[Workflow Router] Task incomplete, retrying (attempt {retry_count + 1})")
        return "orchestrator"
    else:
        print(f"[Workflow Router] Task complete or max retries reached, finishing workflow")
        return "memory_updater"

def route_after_orchestrator(state: RAGState) -> str:
    """Route based on orchestrator's agent selection - supports parallel execution"""
    selected_agents = state.get("selected_agents", [])

    print(f"[Workflow Router] Orchestrator selected agents: {selected_agents if selected_agents else 'None'}")

    # Check if orchestrator provided a direct answer
    if state.get("final_answer") and not selected_agents:
        print(f"[Workflow Router] Direct answer provided, routing to memory_updater")
        return "memory_updater"

    if not selected_agents:
        print(f"[Workflow Router] No agents selected, routing to answer_parser")
        return "answer_parser"

    # Check first item type
    first_item = selected_agents[0]

    # Parallel group detected (nested list)
    if isinstance(first_item, list):
        print(f"[Workflow Router] Parallel group detected: {first_item}, routing to parallel_executor")
        return "parallel_executor"

    # Single agent (string)
    if len(selected_agents) == 1:
        print(f"[Workflow Router] Routing to: {selected_agents[0]}")
        return selected_agents[0]
    else:
        # Multiple agents selected, route to first
        print(f"[Workflow Router] Multiple agents selected, routing to first: {selected_agents[0]}")
        return selected_agents[0]


def parallel_executor_node(state: RAGState) -> RAGState:
    """Execute multiple agents in parallel with explicit field merge"""
    selected = state["selected_agents"][0]

    if not isinstance(selected, list):
        raise ValueError(f"parallel_executor expects list, got {type(selected)}")

    print(f"[Parallel Executor] Running {len(selected)} agents in parallel: {selected}")

    # Agent function map
    agent_map = {
        "character_agent": character_agent,
        "supplementary_agent": supplementary_agent,
        "video_agent": video_agent,
        "audio_agent": audio_agent,
    }

    # Execute in parallel with context propagation
    results = {}
    with ContextThreadPoolExecutor(max_workers=len(selected)) as executor:
        future_to_agent = {
            executor.submit(agent_map[agent], copy.deepcopy(state)): agent
            for agent in selected
        }

        for future in as_completed(future_to_agent):
            agent_name = future_to_agent[future]
            try:
                results[agent_name] = future.result()
                print(f"[Parallel Executor] {agent_name} completed")
            except Exception as e:
                print(f"[Parallel Executor] ERROR: {agent_name} failed - {e}")
                # Store error in agent output field
                output_field = agent_name.replace("_agent", "_output")
                state[output_field] = {"error": str(e), "status": "failed"}

    # Explicit field merge - clear and safe
    for agent_name, agent_state in results.items():
        if agent_name == "character_agent":
            state["character_output"] = agent_state.get("character_output", {})
            state["character_image_metadata"] = agent_state.get("character_image_metadata", {})

        elif agent_name == "supplementary_agent":
            state["supplementary_output"] = agent_state.get("supplementary_output", {})
            state["generated_supplementary"] = agent_state.get("generated_supplementary", {})
            state["supplementary_content_metadata"] = agent_state.get("supplementary_content_metadata", {})

        elif agent_name == "video_agent":
            state["video_output"] = agent_state.get("video_output", {})
            state["generated_video_prompts"] = agent_state.get("generated_video_prompts", {})
            state["video_generation_tasks"] = agent_state.get("video_generation_tasks", [])
            state["video_generation_metadata"] = agent_state.get("video_generation_metadata", {})

        elif agent_name == "audio_agent":
            state["audio_output"] = agent_state.get("audio_output", {})
            state["audio_generation_metadata"] = agent_state.get("audio_generation_metadata", {})
            # Extend asset_urls lists
            if "asset_urls" in agent_state:
                if "asset_urls" not in state:
                    state["asset_urls"] = {}
                for key, items in agent_state["asset_urls"].items():
                    if key not in state["asset_urls"]:
                        state["asset_urls"][key] = []
                    state["asset_urls"][key].extend(items)

        # Merge shared dicts (component_timings, thinking_processes)
        if "component_timings" not in state:
            state["component_timings"] = {}
        state["component_timings"].update(agent_state.get("component_timings", {}))

        if "thinking_processes" not in state:
            state["thinking_processes"] = {}
        state["thinking_processes"].update(agent_state.get("thinking_processes", {}))

    # Remove completed agents from list
    state["selected_agents"] = state["selected_agents"][1:]

    print(f"[Parallel Executor] Completed. Remaining: {state['selected_agents']}")
    return state


def parallel_executor_router(state: RAGState) -> str:
    """Route after parallel execution - check for video monitoring only"""
    # Check video tasks (supplementary now uses blocking pattern)
    video_result = should_monitor_videos(state)
    if video_result == "video_task_monitor":
        return "video_task_monitor"

    # No monitoring needed
    return "answer_parser"


def build_rag_workflow() -> StateGraph:
    """Build the orchestrator-driven workflow with conditional edges"""
    print("[Workflow] Building AI Video Studio workflow graph...")
    
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("memory_manager", memory_manager_agent)
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.add_node("parallel_executor", parallel_executor_node)
    workflow.add_node("script_agent", script_agent)
    workflow.add_node("character_agent", character_agent)
    workflow.add_node("storyboard_agent", storyboard_agent)
    workflow.add_node("supplementary_agent", supplementary_agent)
    workflow.add_node("audio_agent", audio_agent)
    workflow.add_node("video_agent", video_agent)
    workflow.add_node("video_task_monitor", video_task_monitor)
    workflow.add_node("video_editor_agent", video_editor_agent)
    workflow.add_node("answer_parser", answer_parser_agent)
    workflow.add_node("memory_updater", memory_updater_agent)
    
    # Entry point
    workflow.set_entry_point("memory_manager")
    
    # Linear flow to orchestrator
    workflow.add_edge("memory_manager", "orchestrator")
    
    # Conditional routing from orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "parallel_executor": "parallel_executor",
            "script_agent": "script_agent",
            "character_agent": "character_agent",
            "storyboard_agent": "storyboard_agent",
            "supplementary_agent": "supplementary_agent",
            "audio_agent": "audio_agent",
            "video_agent": "video_agent",
            "video_task_monitor": "video_task_monitor",
            "video_editor_agent": "video_editor_agent",
            "answer_parser": "answer_parser",
            "memory_updater": "memory_updater"
        }
    )

    # Parallel executor conditionally routes to monitor or answer_parser
    workflow.add_conditional_edges(
        "parallel_executor",
        parallel_executor_router,
        {
            "video_task_monitor": "video_task_monitor",
            "answer_parser": "answer_parser"
        }
    )

    # Route agents directly to answer_parser
    workflow.add_edge("script_agent", "answer_parser")
    workflow.add_edge("character_agent", "answer_parser")
    workflow.add_edge("storyboard_agent", "answer_parser")
    workflow.add_edge("supplementary_agent", "answer_parser")  # Now uses blocking pattern
    workflow.add_edge("audio_agent", "answer_parser")
    workflow.add_edge("video_editor_agent", "answer_parser")

    # Video agent conditionally routes to monitor or answer_parser
    workflow.add_conditional_edges(
        "video_agent",
        should_monitor_videos,
        {
            "video_task_monitor": "video_task_monitor",
            "answer_parser": "answer_parser"
        }
    )
    workflow.add_edge("video_task_monitor", "answer_parser")

    # Conditional edge from answer parser (retry logic)
    workflow.add_conditional_edges(
        "answer_parser",
        retry_router,
        {
            "orchestrator": "orchestrator",
            "memory_updater": "memory_updater"
        }
    )
    
    # Final step
    workflow.add_edge("memory_updater", END)
    
    compiled_workflow = workflow.compile(checkpointer=MemorySaver())
    print("[Workflow] Workflow compiled successfully")
    
    # Auto-generate graph (only in local environment)
    import os
    deployment_env = os.getenv('DEPLOYMENT_ENV')
    if not deployment_env or deployment_env.lower() == 'local':  # Local development
        try:
            os.makedirs("graphs", exist_ok=True)
            with open("graphs/workflow_graph.png", 'wb') as f:
                f.write(compiled_workflow.get_graph().draw_mermaid_png())
            print("[Workflow] Graph auto-generated: graphs/workflow_graph.png")
        except Exception as e:
            print(f"[Workflow] Could not auto-generate graph: {e}")
    else:
        print(f"[Workflow] Skipping graph generation (deployment: {deployment_env})")
    
    return compiled_workflow

def save_workflow_graph(filename="workflow_graph.png"):
    """Save workflow graph as PNG image"""
    import os
    os.makedirs("graphs", exist_ok=True)
    path = f"graphs/{filename}"
    with open(path, 'wb') as f:
        f.write(rag_workflow.get_graph().draw_mermaid_png())
    print(f"Graph saved: {path}")

# Build the workflow
print("[Workflow] Initializing AI Video Studio workflow...")
rag_workflow = build_rag_workflow()
print("[Workflow] Workflow initialization complete")