"""Redis state manager for session persistence"""

import json
import redis.asyncio as redis
from typing import Dict, Any, Optional, List, Set
from datetime import datetime

class RedisStateManager:
    """
    Manages session state in Redis with automatic TTL.
    Optimized for field-level updates using Redis hashes.
    """
    
    def __init__(self, redis_url: str):
        # Create connection pool with optimized settings
        pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=30,
            decode_responses=True,
            socket_keepalive=True
        )
        self.redis = redis.Redis(connection_pool=pool)
        self.default_ttl = 86400  # 24 hours
        
    async def save_state(self, session_id: str, state: Dict[str, Any]):
        """Save state using Redis hash for efficient field updates

        Args:
            session_id: Session identifier
            state: State to save
        """
        key = f"state_hash:{session_id}"

        # Convert complex values to JSON strings
        encoded_state = {}
        for field, value in state.items():
            if value is not None:  # Skip None values to save space
                encoded_state[field] = json.dumps(value, default=str)

        # Use pipeline for atomic operation
        pipe = self.redis.pipeline()
        pipe.delete(key)  # Clear old state
        if encoded_state:
            pipe.hset(key, mapping=encoded_state)
            pipe.expire(key, self.default_ttl)
        await pipe.execute()
        
    async def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve state from Redis hash"""
        key = f"state_hash:{session_id}"
        
        # Get all fields from hash
        raw_state = await self.redis.hgetall(key)
        if not raw_state:
            return None
            
        # Decode JSON values
        state = {}
        for field, value in raw_state.items():
            try:
                state[field] = json.loads(value)
            except json.JSONDecodeError:
                state[field] = value  # Keep as string if not JSON
        
        return state
        
    async def update_field(self, session_id: str, field: str, value: Any):
        """Update single field without rewriting entire state"""
        key = f"state_hash:{session_id}"
        
        # Encode value and update single field
        encoded_value = json.dumps(value, default=str)
        await self.redis.hset(key, field, encoded_value)
        
        # Refresh TTL
        await self.redis.expire(key, self.default_ttl)
        
    async def append_to_list(self, session_id: str, field: str, item: Any):
        """Append item to list field efficiently"""
        key = f"state_hash:{session_id}"
        
        # Get current list value
        current_value = await self.redis.hget(key, field)
        if current_value:
            current_list = json.loads(current_value)
        else:
            current_list = []
        
        # Append and save back
        current_list.append(item)
        await self.redis.hset(key, field, json.dumps(current_list, default=str))
        
        # Refresh TTL
        await self.redis.expire(key, self.default_ttl)
        
    async def publish_event(self, session_id: str, event: Dict[str, Any]):
        """Publish event to session channel"""
        channel = f"events:{session_id}"
        await self.redis.publish(channel, json.dumps(event, default=str))
        
    async def subscribe_to_session(self, session_id: str):
        """Subscribe to session events"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(f"events:{session_id}")
        return pubsub
        
    async def list_sessions(self) -> List[str]:
        """List all active sessions efficiently"""
        sessions = []
        # Use scan_iter with specific pattern and count hint
        async for key in self.redis.scan_iter("state_hash:*", count=100):
            # Extract session_id from key
            if ":" in key:
                session_id = key.split(":", 1)[1]
                sessions.append(session_id)
        return sessions
        
    async def extend_ttl(self, session_id: str):
        """Extend session TTL"""
        # Update TTL for hash key (new format)
        key = f"state_hash:{session_id}"
        await self.redis.expire(key, self.default_ttl)
        
    async def delete_session(self, session_id: str):
        """Delete session using known key patterns"""
        # Use specific keys instead of wildcard scan
        keys_to_delete = [
            f"session:{session_id}",
            f"state:{session_id}",  # Legacy format
            f"state_hash:{session_id}",  # New hash format
            f"events:{session_id}",
            f"messages:{session_id}"
        ]
        
        # Delete all keys in one command
        await self.redis.delete(*keys_to_delete)
            
    async def save_expert_state(self, expert_id: str, session_id: str, state: Dict[str, Any]):
        """Save expert state to isolated namespace

        Experts use completely separate keys to prevent any interference
        with main workflow state operations.

        Preserves pending_confirmation across executions to maintain context.
        """
        key = f"expert_state:{expert_id}:{session_id}"

        # Get existing state to preserve critical fields
        existing_raw = await self.redis.hgetall(key)

        # Preserve pending_confirmation if not explicitly set
        if existing_raw:
            # Check if there's an existing pending_confirmation we should preserve
            if 'pending_confirmation' in existing_raw:
                try:
                    existing_pending = json.loads(existing_raw['pending_confirmation'])
                    # Only preserve if new state doesn't explicitly set or clear it
                    if 'pending_confirmation' not in state and existing_pending:
                        state['pending_confirmation'] = existing_pending
                        print(f"[RedisState] Preserving pending_confirmation for {expert_id}")
                except (json.JSONDecodeError, KeyError):
                    pass  # Invalid existing data, ignore

        # Convert to JSON for storage
        encoded_state = {}
        for field, value in state.items():
            if value is not None:
                encoded_state[field] = json.dumps(value, default=str)

        # Save merged state
        if encoded_state:
            pipe = self.redis.pipeline()
            # Don't delete - use hset to merge/overwrite fields
            pipe.hset(key, mapping=encoded_state)
            pipe.expire(key, self.default_ttl)
            await pipe.execute()

    async def get_expert_state(self, expert_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Get expert state from isolated namespace"""
        key = f"expert_state:{expert_id}:{session_id}"

        raw_state = await self.redis.hgetall(key)
        if not raw_state:
            return None

        # Decode JSON values
        state = {}
        for field, value in raw_state.items():
            try:
                state[field] = json.loads(value)
            except json.JSONDecodeError:
                state[field] = value

        return state

    async def get_all_expert_states(self, session_id: str) -> Dict[str, Any]:
        """Get all expert states for a session (expert pipeline removed)"""
        return {}

    async def close(self):
        """Close Redis connection"""
        await self.redis.aclose()


async def get_state_from_redis(redis_client, session_id: str) -> Dict[str, Any]:
    """
    Load full state from Redis hash

    Standalone utility function for loading complete session state.
    Used by WebSocket handler for fresh context loading.

    Args:
        redis_client: Redis client instance
        session_id: Session identifier

    Returns:
        Dict containing full session state, or empty dict if not found
    """
    key = f"state_hash:{session_id}"

    raw_state = await redis_client.hgetall(key)
    if not raw_state:
        return {}

    # Decode JSON values
    state = {}
    for field, value in raw_state.items():
        try:
            state[field] = json.loads(value)
        except json.JSONDecodeError:
            state[field] = value

    return state