"""System agents - core video production coordination and workflow management"""

import json
import time
import uuid
import threading
from pathlib import Path
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
from ..base import emit_event

def _annotate_images_thread(state: RAGState, reference_images: list, result_dict: dict):
    """Thread wrapper for image annotation. Stores results in result_dict."""
    try:
        print(f"[Memory Manager Agent] Thread: Annotating {len(reference_images)} reference images...")
        from ...agents.creative.tools_image import get_or_create_annotations

        existing_annotations = state.get("image_annotations", {})
        # Pass state for streaming - ContextThreadPoolExecutor propagates LangGraph context
        annotations = get_or_create_annotations(
            reference_images=reference_images,
            existing_annotations=existing_annotations,
            state=state
        )

        result_dict["annotations"] = annotations
        result_dict["success"] = True
        print(f"[Memory Manager Agent] Thread: Image annotation complete - {len(annotations)} images")
    except Exception as e:
        result_dict["success"] = False
        result_dict["error"] = str(e)
        print(f"[Memory Manager Agent] Thread: Image annotation error: {e}")

def _analyze_documents_thread(state: RAGState, documents: list, result_dict: dict):
    """Thread wrapper for document analysis. Stores results in result_dict."""
    try:
        print(f"[Memory Manager Agent] Thread: Analyzing {len(documents)} document(s)...")
        from ...agents.enterprise.analyzer import analyze_enterprise_documents

        # Analyzer needs state to read enterprise_resources
        # emit_event will handle thread context gracefully (no crash if no LangGraph context)
        analysis_result = analyze_enterprise_documents(state)
        result_dict["analysis"] = analysis_result
        result_dict["success"] = True

        if analysis_result.get("status") == "success":
            print(f"[Memory Manager Agent] Thread: Document analysis complete")
        else:
            print(f"[Memory Manager Agent] Thread: Document analysis status: {analysis_result.get('status')}")
    except Exception as e:
        import traceback
        result_dict["success"] = False
        result_dict["error"] = str(e)
        print(f"[Memory Manager Agent] Thread: Document analysis error: {e}")
        print(f"[Memory Manager Agent] Thread: Traceback:\n{traceback.format_exc()}")

def memory_manager_agent(state: RAGState) -> RAGState:
    """Memory manager with rolling window management"""
    print("[Memory Manager Agent] Starting memory management...")
    print(f"[Memory Manager Agent] DEBUG: State keys present = {list(state.keys())}")
    start_time = time.time()
    
    # Set current agent for event tracking
    state["current_agent"] = "memory_manager"
    
    if "thread_id" not in state:
        state["thread_id"] = str(uuid.uuid4())
    
    if "conversation_history" not in state:
        state["conversation_history"] = []
    
    if "window_memory" not in state:
        state["window_memory"] = []
    
    if "context_summary" not in state:
        state["context_summary"] = ""
    
    if "turn_number" not in state:
        state["turn_number"] = 0
    
    if "thinking_processes" not in state:
        state["thinking_processes"] = {}

    if "component_timings" not in state:
        state["component_timings"] = {}
    
    state["turn_number"] += 1
    state["timestamp"] = datetime.now().isoformat()

    # Reset retry state for new conversation turn
    state["retry_count"] = 0
    if "max_retries" in state:
        del state["max_retries"]  # Force recalculation based on new agent selection

    # Clear monitor/execution outputs from previous turns to prevent cross-turn contamination
    state["video_monitor_output"] = None
    state["storyboard_execution_output"] = None
    state["supplementary_monitor_output"] = None

    # Clear previous attempts for new turn (each turn starts fresh)
    state["previous_attempts"] = []

    # Clear orchestrator retry context for new turn
    # Prevents new user messages from being treated as retries of previous turn
    state["orchestrator_retry_context"] = {}

    # Clear orchestrator outputs from previous turn
    # Prevents stale Workflow Context from misleading LLM on new messages
    state["selected_agents"] = []
    state["execution_plan"] = ""

    if "messages" not in state:
        state["messages"] = []
    
    # Store as simple dict for JSON serialization compatibility
    state["messages"].append({"role": "user", "content": state["user_query"]})
    
    # Note: window_memory will be updated in memory_updater_agent after conversation is added
    # Initialize as empty list if needed, will be populated correctly in memory_updater
    if "window_memory" not in state:
        state["window_memory"] = []
    
    if "metadata" not in state:
        state["metadata"] = {}
    
    if "user_preferences" not in state:
        state["user_preferences"] = {}

    # Initialize expert conversation tracking fields
    if "expert_conversation_history" not in state:
        state["expert_conversation_history"] = {}

    if "expert_window_memory" not in state:
        state["expert_window_memory"] = {}

    if "expert_messages" not in state:
        state["expert_messages"] = []

    if "pending_confirmation" not in state:
        state["pending_confirmation"] = {}

    state["metadata"]["memory_processed"] = True

    # Process uploaded content (images and documents) in parallel when both are present
    reference_images = state.get("reference_images", [])
    enterprise_resources = state.get("enterprise_resources", {})
    documents = enterprise_resources.get("documents", [])

    print(f"[Memory Manager Agent] DEBUG: enterprise_resources = {enterprise_resources}")
    print(f"[Memory Manager Agent] DEBUG: documents count = {len(documents)}")
    print(f"[Memory Manager Agent] DEBUG: has enterprise_agent_output = {bool(state.get('enterprise_agent_output'))}")

    # Check if images need annotation (same logic as image_tools.py line 479-482)
    existing_annotations = state.get("image_annotations", {})
    images_to_annotate = []
    for img in reference_images:
        filename_key = Path(img.split('?')[0]).name
        if filename_key not in existing_annotations:
            images_to_annotate.append(img)

    has_images = bool(images_to_annotate)
    has_documents = bool(documents and not state.get("enterprise_agent_output"))

    if has_images and has_documents:
        # Both present - run in parallel for speed
        print(f"[Memory Manager Agent] Processing {len(images_to_annotate)} images and {len(documents)} documents in PARALLEL...")

        emit_event(
            state,
            "llm_thinking",
            {"content": f"Analyzing {len(documents)} document(s) and {len(images_to_annotate)} image(s)..."},
            agent_name="orchestrator"
        )

        image_result = {}
        document_result = {}

        # Use ContextThreadPoolExecutor to propagate LangGraph context for streaming
        from langchain_core.runnables.config import ContextThreadPoolExecutor
        from concurrent.futures import as_completed

        with ContextThreadPoolExecutor(max_workers=2) as executor:
            image_future = executor.submit(_annotate_images_thread, state, reference_images, image_result)
            doc_future = executor.submit(_analyze_documents_thread, state, documents, document_result)

            # Wait for both to complete
            for future in as_completed([image_future, doc_future]):
                future.result()

        # image_thread = threading.Thread(
        #     target=_annotate_images_thread,
        #     args=(state, reference_images, image_result)
        # )
        # document_thread = threading.Thread(
        #     target=_analyze_documents_thread,
        #     args=(state, documents, document_result)
        # )

        # image_thread.start()
        # document_thread.start()

        # image_thread.join()
        # document_thread.join()

        # Process image results
        if image_result.get("success"):
            state["image_annotations"] = image_result["annotations"]
            print(f"[Memory Manager Agent] Loaded annotations for {len(image_result['annotations'])} images")
        else:
            print(f"[Memory Manager Agent] Warning: Could not annotate images: {image_result.get('error')}")

        # Process document results
        if document_result.get("success"):
            state["enterprise_agent_output"] = document_result["analysis"]
            if document_result["analysis"].get("status") == "success":
                print(f"[Memory Manager Agent] ✓ Enterprise analysis complete - orchestrator will have business context")
            else:
                print(f"[Memory Manager Agent] Enterprise analysis status: {document_result['analysis'].get('status')}")
        else:
            state["enterprise_agent_output"] = {
                "status": "error",
                "error": document_result.get("error"),
                "timestamp": datetime.now().isoformat()
            }

    elif has_images:
        # Only images - run directly without threading overhead
        print(f"[Memory Manager Agent] Found {len(images_to_annotate)} reference images to annotate")

        emit_event(
            state,
            "llm_thinking",
            {"content": f"Analyzing {len(images_to_annotate)} image(s)..."},
            agent_name="orchestrator"
        )

        from ...agents.creative.tools_image import get_or_create_annotations

        try:
            existing_annotations = state.get("image_annotations", {})
            annotations = get_or_create_annotations(
                reference_images=reference_images,
                existing_annotations=existing_annotations,
                state=state
            )

            if annotations:
                state["image_annotations"] = annotations
                print(f"[Memory Manager Agent] Loaded annotations for {len(annotations)} images")
        except Exception as e:
            print(f"[Memory Manager Agent] Warning: Could not annotate images: {e}")

    elif has_documents:
        # Only documents - run directly without threading overhead
        print(f"[Memory Manager Agent] Detected {len(documents)} document(s), analyzing before orchestrator...")

        emit_event(
            state,
            "llm_thinking",
            {"content": f"Analyzing {len(documents)} document(s)..."},
            agent_name="orchestrator"
        )

        try:
            from ...agents.enterprise.analyzer import analyze_enterprise_documents
            analysis_result = analyze_enterprise_documents(state)
            state["enterprise_agent_output"] = analysis_result

            if analysis_result.get("status") == "success":
                print(f"[Memory Manager Agent] ✓ Enterprise analysis complete - orchestrator will have business context")
            else:
                print(f"[Memory Manager Agent] Enterprise analysis status: {analysis_result.get('status')}")
        except Exception as e:
            import traceback
            print(f"[Memory Manager Agent] Error analyzing documents: {e}")
            print(f"[Memory Manager Agent] Traceback:\n{traceback.format_exc()}")
            state["enterprise_agent_output"] = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    window_size = len(state["window_memory"])
    total_conversations = len(state["conversation_history"])
    
    state["thinking_processes"]["memory_manager"] = (
        f"Memory manager initialized. Current conversation history: {total_conversations} conversations. "
        f"Context summary available: {'Yes' if state['context_summary'] else 'No'}. "
        f"Window memory will be updated in memory_updater after processing current turn."
    )
    
    execution_time = time.time() - start_time
    state["component_timings"]["memory_manager"] = execution_time
    print(f"[Memory Manager Agent] Completed in {execution_time:.2f}s - Turn {state['turn_number']}")
    
    return state