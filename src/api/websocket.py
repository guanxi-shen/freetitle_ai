"""WebSocket handler for real-time streaming communication"""

import json
import asyncio
import time
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
import redis.asyncio as redis

from ..core.streaming import stream_workflow
from .redis_state import RedisStateManager, get_state_from_redis

# Flag to enable non-blocking WebSocket sends to prevent event buffering
# This addresses the issue where WebSocket sends can block when multiple workflows
# run in parallel, causing events to buffer until one workflow completes
# Set to False to revert to blocking sends if this approach causes issues
ENABLE_NONBLOCKING_WEBSOCKET_SENDS = True

# Sequential event queue flag - controls event ordering vs parallel execution tradeoff
# When True: All events go through a single queue, ensuring strict order but serializing workflows
# When False: Events stream directly, allowing true parallel execution of expert and main workflows
#
# History: Sequential queue was added to fix potential event ordering issues from non-blocking sends,
# but it reintroduced workflow blocking - expert agents would block the main workflow despite running
# as separate asyncio tasks. Setting to False allows true parallel execution.
#
# Tradeoff: May cause minor event ordering issues within a single workflow (e.g., token streaming),
# but enables expert agents to run truly in parallel without blocking main workflow.
# Frontend handles async events robustly using timestamps and execution IDs.
ENABLE_SEQUENTIAL_EVENT_QUEUE = False  # Disabled to allow true parallel expert execution

class WebSocketHandler:
    """Handles WebSocket connections and streaming for workflow execution"""

    def __init__(self, websocket: WebSocket, session_id: str, redis_client: redis.Redis):
        self.websocket = websocket
        self.session_id = session_id
        self.redis = redis_client
        self.running_workflow = None  # Track main creative workflow task
        self.event_queue = None  # Will be initialized if sequential queue is enabled
        self.sender_task = None  # Task for sequential event sending
        self.is_closing = False  # Track if WebSocket is closing

    async def _ensure_state_ready(
        self,
        query: str,
        reference_images: Optional[List[str]] = None,
        tool_selections: Optional[Dict[str, List[str]]] = None,
        expert_selections: Optional[List[str]] = None,
        enable_validation: Optional[bool] = None,
        workflow_mode: Optional[str] = None
    ):
        """Ensure state exists in Redis with current turn's query before launching workflows

        Keeps it simple: updates essential fields if state exists, creates minimal state if not.
        Workflows will build remaining fields as needed.
        """

        # Get existing state from Redis (may be from resume_state or previous turn)
        existing_state = await get_state_from_redis(self.redis, self.session_id)

        if existing_state:
            # State exists - just update fields for current turn
            print(f"[WebSocket] Updating state for session {self.session_id} (turn {existing_state.get('turn_number', 0) + 1})")

            # Update essential fields using Redis hash operations
            updates = {
                "user_query": json.dumps(query),
                "turn_number": json.dumps(existing_state.get("turn_number", 0) + 1),
                "final_answer": json.dumps(None)  # Clear previous answer
            }

            if reference_images:
                updates["reference_images"] = json.dumps(reference_images)

            if tool_selections or enable_validation is not None or workflow_mode is not None:
                user_prefs = existing_state.get("user_preferences", {})
                if tool_selections:
                    user_prefs["tool_selections"] = tool_selections
                if enable_validation is not None:
                    user_prefs["enable_validation"] = enable_validation
                if workflow_mode is not None:
                    user_prefs["workflow_mode"] = workflow_mode
                updates["user_preferences"] = json.dumps(user_prefs)

            if expert_selections is not None:
                updates["expert_selections"] = json.dumps(expert_selections)

            # Update fields atomically
            await self.redis.hset(f"state_hash:{self.session_id}", mapping=updates)

        else:
            # New session - create full initial state (must match streaming.py structure)
            # NOTE: Do NOT initialize enterprise_resources or enterprise_agent_output here
            # They are created by upload endpoint when documents are uploaded
            print(f"[WebSocket] Creating new state for session {self.session_id}")

            minimal_state = {
                "user_query": query,
                "session_id": self.session_id,
                "thread_id": self.session_id,
                "messages": [],
                "turn_number": 1,
                "reference_images": reference_images or [],
                "user_preferences": {
                    "tool_selections": tool_selections or {
                        "video_agent": ["google_veo_i2v"]
                    },
                    "workflow_mode": workflow_mode or "step-by-step",
                    "enable_validation": enable_validation or False
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
                "video_generation_metadata": {}
            }

            # Save minimal state - streaming.py will build the rest
            redis_url = os.getenv("REDIS_URL_TEST", os.getenv("REDIS_URL", "redis://localhost:6379"))
            state_manager = RedisStateManager(redis_url)
            await state_manager.save_state(self.session_id, minimal_state)
            await state_manager.close()

        print(f"[WebSocket] State ready in Redis for session {self.session_id}")

    async def _event_sender_loop(self):
        """Sequential event sender that maintains order"""
        try:
            while True:
                event = await self.event_queue.get()
                try:
                    # Send to WebSocket
                    await self.websocket.send_text(json.dumps(event, default=str))

                    # Also publish to Redis for other consumers
                    await self.redis.publish(
                        f"events:{self.session_id}",
                        json.dumps(event, default=str)
                    )
                except Exception as e:
                    print(f"[WebSocket] Error in sequential send: {str(e)}")
        except asyncio.CancelledError:
            print(f"[WebSocket] Event sender loop cancelled for session {self.session_id}")
            raise

    async def handle(self):
        """Main WebSocket message handler"""
        try:
            # Initialize event queue if sequential sending is enabled
            if ENABLE_SEQUENTIAL_EVENT_QUEUE:
                self.event_queue = asyncio.Queue()
                self.sender_task = asyncio.create_task(self._event_sender_loop())
                print(f"[WebSocket] Sequential event queue initialized for session {self.session_id}")

            while True:
                # Receive message from client
                raw_data = await self.websocket.receive_text()
                print(f"[WebSocket] Received raw message: {raw_data[:200]}...")
                data = json.loads(raw_data)
                print(f"[WebSocket] Parsed message type: {data.get('type', 'UNKNOWN')}")

                # Route based on message type
                if data["type"] == "execute":
                    # Check if resume_state provided
                    resume_state = data.get("resume_state")
                    if resume_state:
                        # Check if state has saved_sessions URLs and regenerate if needed
                        from ..storage.gcs_utils import detect_saved_session_name, regenerate_session_urls

                        save_name = detect_saved_session_name(resume_state)
                        if save_name:
                            print(f"[WebSocket] Detected saved session '{save_name}', regenerating URLs...")
                            resume_state = regenerate_session_urls(resume_state, save_name)

                        # Load the resume state into Redis using hash format
                        key = f"state_hash:{self.session_id}"

                        # Convert state to hash fields
                        encoded_state = {}
                        for field, value in resume_state.items():
                            if value is not None:
                                encoded_state[field] = json.dumps(value, default=str)

                        # Store using pipeline for atomicity
                        pipe = self.redis.pipeline()
                        pipe.delete(key)
                        if encoded_state:
                            pipe.hset(key, mapping=encoded_state)
                            pipe.expire(key, 7200)  # 2 hours TTL
                        await pipe.execute()

                        print(f"[WebSocket] Loaded resume state for session {self.session_id}")

                    # Get active experts from request
                    print(f"[WebSocket] Getting active_experts from data: {type(data)}")
                    active_experts = data.get("active_experts", [])
                    print(f"[WebSocket] Getting query from data")
                    query = data["query"]
                    print(f"[WebSocket] Getting tool_selections from data")
                    tool_selections = data.get("tool_selections", {})
                    enable_validation = data.get("enable_validation", False)
                    workflow_mode = data.get("workflow_mode", "step-by-step")

                    print(f"[WebSocket] ========== PARALLEL EXECUTION START ==========")
                    print(f"[WebSocket] Query: {query[:100]}...")
                    print(f"[WebSocket] Active experts: {active_experts}")
                    print(f"[WebSocket] Tool selections: {tool_selections}")
                    print(f"[WebSocket] Validation enabled: {enable_validation}")
                    print(f"[WebSocket] Workflow mode: {workflow_mode}")

                    # Prepare state before launching any workflows
                    await self._ensure_state_ready(
                        query=query,
                        reference_images=data.get("reference_images", []),
                        tool_selections=tool_selections,
                        expert_selections=active_experts,
                        enable_validation=enable_validation,
                        workflow_mode=workflow_mode
                    )

                    # Launch Creative workflow as independent task (non-blocking)
                    print(f"[WebSocket] Launching Creative workflow (parallel mode)...")
                    creative_task = asyncio.create_task(
                        self.execute_workflow(
                            query=query,
                            reference_images=data.get("reference_images", []),
                            tool_selections=tool_selections,
                            expert_selections=active_experts
                        )
                    )

                    # Track workflow task for interrupt support
                    self.running_workflow = creative_task

                    # Add async completion handler for creative workflow
                    async def handle_creative_completion():
                        try:
                            await creative_task
                            print(f"[WebSocket] ✓ Creative workflow completed successfully")
                        except asyncio.CancelledError:
                            print(f"[WebSocket] Creative workflow cancelled by user")
                        except Exception as e:
                            import traceback
                            print(f"[WebSocket] Creative workflow error: {str(e)}")
                            print(f"[WebSocket] Traceback:\n{traceback.format_exc()}")
                            # Send error event but don't block
                            try:
                                await self.send_event({
                                    "type": "error",
                                    "message": f"Creative workflow error: {str(e)}"
                                })
                            except:
                                pass  # Ignore send errors in completion handler

                    # Schedule completion handler without blocking
                    asyncio.create_task(handle_creative_completion())

                    print(f"[WebSocket] ========== WORKFLOW EXECUTION INITIATED ==========")
                elif data["type"] == "get_state":
                    await self.send_current_state()
                elif data["type"] == "interrupt":
                    # Cancel all running workflows and experts
                    print(f"[WebSocket] Interrupt requested for session {self.session_id}")

                    cancelled_count = 0

                    # Cancel main creative workflow
                    if self.running_workflow and not self.running_workflow.done():
                        print(f"[WebSocket] Cancelling creative workflow")
                        self.running_workflow.cancel()
                        cancelled_count += 1

                    print(f"[WebSocket] Interrupted {cancelled_count} running task(s)")

                    await self.send_event({
                        "type": "workflow_interrupted",
                        "cancelled_count": cancelled_count,
                        "timestamp": datetime.now().isoformat()
                    })
                elif data["type"] == "regenerate_asset":
                    # Regenerate a specific asset (storyboard frame or video)
                    asset_type = data.get("asset_type")
                    asset_id = data.get("asset_id")
                    revision_notes = data.get("revision_notes")

                    print(f"[WebSocket] Regenerate request: {asset_type} - {asset_id}")
                    if revision_notes:
                        print(f"[WebSocket] Revision notes: {revision_notes[:100]}...")

                    # Launch regeneration as background task
                    asyncio.create_task(
                        self.regenerate_asset(asset_type, asset_id, revision_notes)
                    )
                elif data["type"] == "ping":
                    await self.send_event({"type": "pong"})
                else:
                    await self.send_event({
                        "type": "error",
                        "message": f"Unknown message type: {data['type']}"
                    })

        except WebSocketDisconnect as e:
            self.is_closing = True
            print(f"[WebSocket] Client disconnected (code={e.code}, session={self.session_id})")
            # Clean up sender task if it exists
            if self.sender_task:
                self.sender_task.cancel()
                try:
                    await self.sender_task
                except asyncio.CancelledError:
                    pass
                print(f"[WebSocket] Event sender task cleaned up for session {self.session_id}")

        except json.JSONDecodeError as e:
            await self.send_event({
                "type": "error",
                "message": f"Invalid JSON: {str(e)}"
            })
        except Exception as e:
            self.is_closing = True
            import traceback
            print(f"[WebSocket] Handler error: {str(e)}")
            print(f"[WebSocket] Traceback:\n{traceback.format_exc()}")
            # Clean up sender task on error
            if self.sender_task:
                self.sender_task.cancel()
            raise
                
    async def execute_workflow(self, query: str, reference_images: Optional[list] = None, tool_selections: Optional[dict] = None, expert_selections: Optional[list] = None):
        """Execute workflow with streaming"""
        try:
            # Send workflow started event
            await self.send_event({
                "type": "workflow_started",
                "timestamp": datetime.now().isoformat()
            })

            # Stream workflow execution
            # Note: stream_workflow handles streaming via callback, not as a generator
            await stream_workflow(
                query=query,
                session_id=self.session_id,
                reference_images=reference_images or [],
                tool_selections=tool_selections or {},
                expert_selections=expert_selections or [],
                stream_callback=self.send_event
            )
                
            # Send completion event
            await self.send_event({
                "type": "workflow_complete",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            import traceback
            print(f"[WebSocket] Workflow execution error: {str(e)}")
            print(f"[WebSocket] Traceback:\n{traceback.format_exc()}")
            await self.send_event({
                "type": "error",
                "message": str(e)
            })
            
    async def send_event(self, event: Dict[str, Any]):
        """Send event to WebSocket client"""
        if self.is_closing:
            return

        try:
            if ENABLE_SEQUENTIAL_EVENT_QUEUE and self.event_queue:
                # Queue events for sequential sending
                await self.event_queue.put(event)
            elif ENABLE_NONBLOCKING_WEBSOCKET_SENDS:
                # Create task for WebSocket send to prevent blocking parallel workflows
                # This allows events from multiple workflows to be sent without waiting
                async def send_nowait():
                    if self.is_closing:
                        return
                    try:
                        await self.websocket.send_text(json.dumps(event, default=str))
                    except Exception as e:
                        print(f"[WebSocket] Error in non-blocking send: {str(e)}")

                asyncio.create_task(send_nowait())

                # Redis publish can also be made non-blocking for non-critical events
                async def publish_nowait():
                    try:
                        await self.redis.publish(
                            f"events:{self.session_id}",
                            json.dumps(event, default=str)
                        )
                    except Exception:
                        pass  # Redis publish is non-critical

                asyncio.create_task(publish_nowait())
            else:
                # Original blocking implementation
                await self.websocket.send_text(json.dumps(event, default=str))

                # Also publish to Redis for other consumers
                await self.redis.publish(
                    f"events:{self.session_id}",
                    json.dumps(event, default=str)
                )
        except Exception as e:
            print(f"[WebSocket] Error sending event: {str(e)}")
            
    async def send_current_state(self):
        """Send current session state from hash storage"""
        try:
            key = f"state_hash:{self.session_id}"

            # Get all fields from hash
            raw_state = await self.redis.hgetall(key)

            if raw_state:
                # Decode JSON values
                state = {}
                for field, value in raw_state.items():
                    try:
                        state[field] = json.loads(value)
                    except json.JSONDecodeError:
                        state[field] = value

                # Return gs:// paths directly - client will resolve to signed URLs on-demand
                print(f"[WebSocket] Sending state with gs:// paths (client resolves URLs on-demand)")

                await self.send_event({
                    "type": "state",
                    "data": state
                })
            else:
                await self.send_event({
                    "type": "state",
                    "data": {}
                })
        except Exception as e:
            await self.send_event({
                "type": "error",
                "message": f"Error getting state: {str(e)}"
            })


    # =========================================================================
    # ASSET REGENERATION
    # =========================================================================

    async def regenerate_asset(
        self,
        asset_type: str,
        asset_id: str,
        revision_notes: Optional[str] = None
    ):
        """
        Regenerate a storyboard frame or video.

        Args:
            asset_type: "storyboard" or "video"
            asset_id: For storyboard: filename like "sc01_sh01_fr1.png"
                      For video: scene_shot key like "1_1"
            revision_notes: Optional revision instructions. If None, uses same params (direct mode).
                           If provided, routes to agent (agent-guided mode).
        """
        mode = "agent" if revision_notes else "direct"
        print(f"[WebSocket] Starting {asset_type} regeneration ({mode} mode): {asset_id}")

        try:
            # Emit regeneration started event
            await self.send_event({
                "type": "regeneration_started",
                "asset_type": asset_type,
                "asset_id": asset_id,
                "mode": mode,
                "timestamp": datetime.now().isoformat()
            })

            # Load current state from Redis
            state = await get_state_from_redis(self.redis, self.session_id)
            if not state:
                raise ValueError(f"No state found for session {self.session_id}")

            # Route based on mode and asset type
            if revision_notes:
                result = await self._regenerate_with_agent(
                    state, asset_type, asset_id, revision_notes
                )
            else:
                if asset_type == "storyboard":
                    result = await self._regenerate_storyboard_direct(state, asset_id)
                elif asset_type == "video":
                    result = await self._regenerate_video_direct(state, asset_id)
                else:
                    raise ValueError(f"Unknown asset type: {asset_type}")

            if result.get("success"):
                print(f"[WebSocket] Regeneration successful: {asset_id}")
                # Emit appropriate event based on asset type
                if asset_type == "storyboard":
                    await self.send_event({
                        "type": "storyboard_frame_regenerated",
                        "data": result.get("data", {}),
                        "timestamp": datetime.now().isoformat()
                    })
                # Video events are emitted by existing video_task_monitor flow
            else:
                raise ValueError(result.get("error", "Regeneration failed"))

        except Exception as e:
            import traceback
            print(f"[WebSocket] Regeneration error: {str(e)}")
            print(f"[WebSocket] Traceback:\n{traceback.format_exc()}")
            await self.send_event({
                "type": "regeneration_failed",
                "asset_type": asset_type,
                "asset_id": asset_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    async def _regenerate_storyboard_direct(
        self,
        state: Dict[str, Any],
        filename: str
    ) -> Dict[str, Any]:
        """
        Direct storyboard regeneration using stored params.
        Calls the generation tool directly without agent involvement.
        """
        print(f"[WebSocket] Direct storyboard regeneration: {filename}")

        # Get metadata from state
        metadata = state.get("storyboard_frame_metadata", {}).get(filename)
        if not metadata:
            return {"success": False, "error": f"No metadata found for {filename}"}

        params = metadata.get("params", {})
        if not params:
            return {"success": False, "error": f"No params stored for {filename}"}

        # Extract generation parameters
        scene = params.get("scene_number")
        shot = params.get("shot_number")
        frame = params.get("frame_number", 1)
        prompt = params.get("prompt")
        tool = params.get("tool", "nano_banana")
        aspect_ratio = params.get("aspect_ratio", "horizontal")
        reference_images = params.get("reference_images", [])

        if not all([scene, shot, prompt]):
            return {"success": False, "error": f"Missing required params: scene={scene}, shot={shot}, prompt={prompt[:50] if prompt else None}"}

        print(f"[WebSocket] Regenerating S{scene} Shot{shot} F{frame} with {tool}")

        # Import and call tool directly (in thread pool to avoid blocking)
        from ..agents.creative.tools_image import generate_or_edit_frame_sync

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: generate_or_edit_frame_sync(
                scene=scene,
                shot=shot,
                frame=frame,
                prompt=prompt,
                reference_images=reference_images if reference_images else None,
                tool=tool,
                aspect_ratio=aspect_ratio,
                session_id=state.get("session_id", self.session_id)
            )
        )

        if result.get("success"):
            # Update state metadata
            new_metadata = {
                "scene": scene,
                "shot": shot,
                "frame": frame,
                "path": result.get("gcs_url"),
                "success": True,
                "prompt": prompt,
                "tool": tool,
                "timestamp": datetime.now().isoformat(),
                "params": params,
                "orchestrator_config": metadata.get("orchestrator_config", {})
            }

            # Update Redis state
            state["storyboard_frame_metadata"][filename] = new_metadata
            await self.redis.hset(
                f"state_hash:{self.session_id}",
                "storyboard_frame_metadata",
                json.dumps(state["storyboard_frame_metadata"])
            )

            return {
                "success": True,
                "data": {
                    "filename": filename,
                    "path": result.get("gcs_url"),
                    "scene": scene,
                    "shot": shot,
                    "frame": frame
                }
            }
        else:
            return {"success": False, "error": result.get("error", "Generation failed")}

    async def _regenerate_video_direct(
        self,
        state: Dict[str, Any],
        task_key: str
    ) -> Dict[str, Any]:
        """
        Direct video regeneration using stored params.
        Submits a new video task with the same parameters.
        """
        print(f"[WebSocket] Direct video regeneration: {task_key}")

        # Parse scene_shot key
        try:
            scene, shot = map(int, task_key.split("_"))
        except ValueError:
            return {"success": False, "error": f"Invalid task key format: {task_key}"}

        # Find metadata from video_generation_metadata
        metadata = None
        video_meta = state.get("video_generation_metadata", {})
        for key, meta in video_meta.items():
            if meta.get("scene") == scene and meta.get("shot") == shot:
                metadata = meta
                break

        # Fallback to video_generation_tasks
        if not metadata:
            for task in state.get("video_generation_tasks", []):
                if task.get("scene") == scene and task.get("shot") == shot:
                    metadata = task
                    break

        if not metadata:
            return {"success": False, "error": f"No metadata found for scene {scene}, shot {shot}"}

        # Extract parameters
        tool_name = metadata.get("tool_used", "google_veo_i2v")
        params = metadata.get("params", {})
        generation_prompt = params.get("generation_prompt")
        start_frame_path = params.get("start_frame_path")
        end_frame_path = params.get("end_frame_path")
        duration = params.get("duration", 5)
        aspect_ratio = params.get("aspect_ratio", "horizontal")

        if not generation_prompt or not start_frame_path:
            return {"success": False, "error": f"Missing required params for video regeneration"}

        print(f"[WebSocket] Regenerating video S{scene} Shot{shot} with {tool_name}")

        # Import video tools
        from ..agents.creative.tools_video import VIDEO_TOOLS

        if tool_name not in VIDEO_TOOLS:
            return {"success": False, "error": f"Unknown video tool: {tool_name}"}

        tool_func = VIDEO_TOOLS[tool_name]

        # Submit video task (in thread pool)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: tool_func(
                generation_prompt=generation_prompt,
                start_frame_path=start_frame_path,
                end_frame_path=end_frame_path,
                duration=duration,
                session_id=state.get("session_id", self.session_id),
                aspect_ratio=aspect_ratio,
                scene=scene,
                shot=shot
            )
        )

        if result.get("success"):
            task_id = result.get("task_id")

            # Add to video_generation_tasks for monitoring
            new_task = {
                "task_id": task_id,
                "scene": scene,
                "shot": shot,
                "status": "submitted",
                "tool_used": tool_name,
                "params": params,
                "submitted_at": datetime.now().isoformat(),
                "is_regeneration": True
            }

            # Get existing tasks and append
            existing_tasks = state.get("video_generation_tasks", [])
            existing_tasks.append(new_task)
            state["video_generation_tasks"] = existing_tasks

            # Update Redis
            await self.redis.hset(
                f"state_hash:{self.session_id}",
                "video_generation_tasks",
                json.dumps(existing_tasks)
            )

            # Emit video_task_submitted event (UI already handles this)
            await self.send_event({
                "type": "video_task_submitted",
                "data": {
                    "task_id": task_id,
                    "scene": scene,
                    "shot": shot,
                    "tool_used": tool_name,
                    "is_regeneration": True
                },
                "timestamp": datetime.now().isoformat()
            })

            return {
                "success": True,
                "data": {
                    "task_id": task_id,
                    "scene": scene,
                    "shot": shot,
                    "tool_used": tool_name,
                    "status": "submitted"
                }
            }
        else:
            return {"success": False, "error": result.get("error", "Video submission failed")}

    async def _regenerate_with_agent(
        self,
        state: Dict[str, Any],
        asset_type: str,
        asset_id: str,
        revision_notes: str
    ) -> Dict[str, Any]:
        """
        Agent-guided regeneration with revision notes.
        Routes to the appropriate agent with revision context.
        """
        print(f"[WebSocket] Agent-guided {asset_type} regeneration: {asset_id}")
        print(f"[WebSocket] Revision notes: {revision_notes}")

        if asset_type == "storyboard":
            return await self._regenerate_storyboard_with_agent(state, asset_id, revision_notes)
        elif asset_type == "video":
            return await self._regenerate_video_with_agent(state, asset_id, revision_notes)
        else:
            return {"success": False, "error": f"Unknown asset type: {asset_type}"}

    async def _regenerate_storyboard_with_agent(
        self,
        state: Dict[str, Any],
        filename: str,
        revision_notes: str
    ) -> Dict[str, Any]:
        """
        Regenerate storyboard frame using agent with revision notes.
        Uses the full system prompt + revision notes as workflow instruction.
        """
        import re

        # Parse filename to get scene/shot/frame
        match = re.match(r'sc(\d+)_sh(\d+)_fr(\d+)', filename)
        if not match:
            return {"success": False, "error": f"Invalid filename format: {filename}"}

        scene, shot, frame = int(match.group(1)), int(match.group(2)), int(match.group(3))
        print(f"[WebSocket] Agent regeneration for S{scene} Shot{shot} F{frame}")

        # Get existing metadata for context
        existing_meta = state.get("storyboard_frame_metadata", {}).get(filename, {})
        existing_params = existing_meta.get("params", {})
        orchestrator_config = existing_meta.get("orchestrator_config", {})

        # Build revision instruction
        revision_instruction = f"""REGENERATION REQUEST for Scene {scene}, Shot {shot}, Frame {frame}

USER REVISION NOTES:
{revision_notes}

ORIGINAL GENERATION CONTEXT:
- Previous prompt: {existing_params.get('prompt', 'N/A')}
- Tool used: {existing_params.get('tool', 'nano_banana')}

Apply the revision notes while maintaining visual consistency with the project style.
Generate ONLY this specific frame - do not process other shots."""

        # Import and call shot processor
        from ..agents.creative.agent_storyboard_shot_processor import process_storyboard_shots

        # Get tool from existing metadata
        selected_tools = [existing_params.get("tool", "nano_banana")]
        reference_images = orchestrator_config.get("reference_images", [])

        # Run in thread pool
        loop = asyncio.get_event_loop()
        function_results_collector = []

        try:
            result = await loop.run_in_executor(
                None,
                lambda: process_storyboard_shots(
                    shots=[{"scene": scene, "shot": shot}],
                    state=state,
                    storyboard_instruction=revision_instruction,
                    reference_images=reference_images,
                    function_results_collector=function_results_collector,
                    selected_tools=selected_tools
                )
            )
        except Exception as e:
            return {"success": False, "error": f"Agent processing failed: {str(e)}"}

        # Check if frame was regenerated
        updated_meta = state.get("storyboard_frame_metadata", {}).get(filename, {})
        if updated_meta.get("success"):
            # Update Redis with new metadata
            await self.redis.hset(
                f"state_hash:{self.session_id}",
                "storyboard_frame_metadata",
                json.dumps(state["storyboard_frame_metadata"])
            )

            return {
                "success": True,
                "data": {
                    "filename": filename,
                    "path": updated_meta.get("path"),
                    "scene": scene,
                    "shot": shot,
                    "frame": frame
                }
            }
        else:
            return {"success": False, "error": "Agent regeneration did not produce a successful frame"}

    async def _regenerate_video_with_agent(
        self,
        state: Dict[str, Any],
        task_key: str,
        revision_notes: str
    ) -> Dict[str, Any]:
        """
        Regenerate video using agent with revision notes.
        Routes to video shot processor with revision context.
        """
        # Parse scene_shot key
        try:
            scene, shot = map(int, task_key.split("_"))
        except ValueError:
            return {"success": False, "error": f"Invalid task key format: {task_key}"}

        print(f"[WebSocket] Agent video regeneration for S{scene} Shot{shot}")

        # Find existing metadata
        metadata = None
        for key, meta in state.get("video_generation_metadata", {}).items():
            if meta.get("scene") == scene and meta.get("shot") == shot:
                metadata = meta
                break

        if not metadata:
            for task in state.get("video_generation_tasks", []):
                if task.get("scene") == scene and task.get("shot") == shot:
                    metadata = task
                    break

        existing_params = metadata.get("params", {}) if metadata else {}
        existing_prompt = existing_params.get("generation_prompt", {})

        # Build revision instruction
        revision_instruction = f"""REGENERATION REQUEST for Scene {scene}, Shot {shot}

USER REVISION NOTES:
{revision_notes}

ORIGINAL GENERATION CONTEXT:
- Previous motion prompt: {existing_prompt.get('motion', 'N/A') if isinstance(existing_prompt, dict) else existing_prompt}
- Tool used: {metadata.get('tool_used', 'google_veo_i2v') if metadata else 'N/A'}

Apply the revision notes to create an improved video."""

        # Inject revision instruction into state
        state["video_regeneration_instruction"] = revision_instruction

        # Import and call video shot processor
        from ..agents.creative.agent_video_shot_processor import process_single_shot

        # Get storyboard data for this shot
        storyboard_frames = []
        for fname, meta in state.get("storyboard_frame_metadata", {}).items():
            if meta.get("scene") == scene and meta.get("shot") == shot and meta.get("success"):
                storyboard_frames.append({
                    "url": meta.get("path"),
                    "scene": scene,
                    "shot": shot,
                    "frame": meta.get("frame", 1),
                    "filename": fname
                })

        if not storyboard_frames:
            return {"success": False, "error": f"No storyboard frames found for S{scene} Shot{shot}"}

        # Get available tools
        available_tools = state.get("user_preferences", {}).get("tool_selections", {}).get("video_agent", ["google_veo_i2v"])

        # Run in thread pool
        loop = asyncio.get_event_loop()
        function_results_collector = []

        try:
            result = await loop.run_in_executor(
                None,
                lambda: process_single_shot(
                    scene=scene,
                    shot=shot,
                    state=state,
                    available_tools=available_tools,
                    storyboard_data={"frames": storyboard_frames},
                    function_results_collector=function_results_collector
                )
            )
        except Exception as e:
            return {"success": False, "error": f"Agent processing failed: {str(e)}"}

        if result.get("success"):
            task_id = result.get("task_id")

            # Add to video_generation_tasks for monitoring
            new_task = {
                "task_id": task_id,
                "scene": scene,
                "shot": shot,
                "status": "submitted",
                "tool_used": result.get("tool_used", available_tools[0]),
                "params": result.get("params", {}),
                "submitted_at": datetime.now().isoformat(),
                "is_regeneration": True
            }

            existing_tasks = state.get("video_generation_tasks", [])
            existing_tasks.append(new_task)
            state["video_generation_tasks"] = existing_tasks

            # Update Redis
            await self.redis.hset(
                f"state_hash:{self.session_id}",
                "video_generation_tasks",
                json.dumps(existing_tasks)
            )

            # Emit video_task_submitted event
            await self.send_event({
                "type": "video_task_submitted",
                "data": {
                    "task_id": task_id,
                    "scene": scene,
                    "shot": shot,
                    "tool_used": result.get("tool_used"),
                    "is_regeneration": True
                },
                "timestamp": datetime.now().isoformat()
            })

            return {
                "success": True,
                "data": {
                    "task_id": task_id,
                    "scene": scene,
                    "shot": shot,
                    "tool_used": result.get("tool_used"),
                    "status": "submitted"
                }
            }
        else:
            return {"success": False, "error": result.get("error", "Agent video regeneration failed")}