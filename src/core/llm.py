"""Gemini LLM integration for multi-agent video generation"""

import json
import logging
import random
import re
import httpx
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain_core.runnables.config import ContextThreadPoolExecutor
from google import genai
from google.genai import types
from .config import PROJECT_ID, CREDENTIALS, MCP_SERVER_URL, MCP_REQUEST_TIMEOUT, LOCATION
from fastmcp import Client
from src.schemas import get_schema

logger = logging.getLogger(__name__)


class GeminiVertexLLM(LLM):
    """Gemini LLM implementation using Google Vertex AI platform"""

    model_name: str = "gemini-3-pro-preview"
    locations: List[str] = [LOCATION]  # Use same location as cache creation for compatibility
    gemini_configs: Dict = {
        'max_output_tokens': 2048,
        'temperature': 1,
    }
    enable_thinking: bool = True
    thinking_budget_tokens: int = -1
    thinking_level: str = "high"  # For Gemini 3: "high" or "low"
    enable_images: bool = False
    system_instruction: str = None
    cached_content_name: str = None  # Gemini cache resource name for context caching
    function_declarations: List[Dict] = None
    tools: List = None  # For direct Python function tools
    tool_config: Dict = None

    def __init__(self, **kwargs):
        """Initialize with custom parameters"""
        super().__init__()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def setup_gemini(self):
        """Initialize Gemini client with load balancing across regions"""
        # Gemini 3 requires global location, otherwise use load balancing
        if "gemini-3" in self.model_name.lower():
            location = "global"
        else:
            location = random.choice(self.locations)
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=location,
            credentials=CREDENTIALS
        )
        return client

    def count_tokens(self, content, client=None):
        """Count tokens using Gemini's API for accurate measurement"""
        if client is None:
            client = self.setup_gemini()

        try:
            result = client.models.count_tokens(
                model=self.model_name,
                contents=content
            )
            return result.total_tokens
        except Exception as e:
            print(f"[Token Counter] Error counting tokens: {e}")
            # Fallback to character-based estimation
            content_str = str(content) if not isinstance(content, str) else content
            return len(content_str) // 4
    
    # REMOVED: Async execution methods for web search (not needed for manual function calling)
    # Manual function calling now uses ContextThreadPoolExecutor directly with registered tools

    def _handle_automatic_function_response(self, response):
        """Handle response from automatic function calling
        
        Extracts function call history and results from automatic_function_calling_history
        """
        thinking_text = ""
        response_text = ""
        function_calls_made = []
        
        # Extract final response (thinking and text)
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'thought') and part.thought and self.enable_thinking:
                        thinking_text = part.text or ""
                    elif hasattr(part, 'text') and part.text:
                        response_text = part.text
        
        # Fallback to response.text if needed
        if not response_text and hasattr(response, 'text'):
            response_text = response.text or ""
        
        # Clean JSON wrapper if present
        if response_text.strip().startswith('```json'):
            response_text = response_text.strip()[7:-3].strip()
        
        # Extract function call history from automatic_function_calling_history
        if hasattr(response, 'automatic_function_calling_history') and response.automatic_function_calling_history:
            history = response.automatic_function_calling_history
            
            # Extract function calls from history for debugging
            for item in history:
                if hasattr(item, 'role') and item.role == 'model' and hasattr(item, 'parts'):
                    for part in item.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            call_info = {
                                "name": fc.name if hasattr(fc, 'name') else "unknown",
                                "args": dict(fc.args) if hasattr(fc, 'args') else {}
                            }
                            function_calls_made.append(call_info)
            
            # Note: We don't process function responses since we rely on filesystem scanning
            # The SDK has limitations with automatic function calling responses
            # Function calls are tracked for debugging only
        
        # Build response JSON
        result = {
            "content": [
                {"thinking": thinking_text},
                {"text": response_text}
            ]
        }
        
        # Add function calls if any were found
        if function_calls_made:
            result["function_calls"] = function_calls_made
        
        return json.dumps(result, indent=2)
    
    def _handle_standard_response(self, response):
        """Handle standard response, including automatic function call results"""
        thinking_text = ""
        response_text = ""
        function_calls_made = []
        
        # Extract thinking, text, and function calls from Parts
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'thought') and part.thought and self.enable_thinking:
                        # This is thinking content
                        thinking_text = part.text or ""
                    elif hasattr(part, 'text') and part.text:
                        # This is regular response content
                        response_text = part.text
                    elif hasattr(part, 'function_call'):
                        # This is a function call made during automatic function calling
                        fc = part.function_call
                        function_call_info = {
                            "name": fc.name if hasattr(fc, 'name') else "unknown",
                            "args": dict(fc.args) if hasattr(fc, 'args') else {}
                        }
                        # With automatic function calling, the SDK executes but doesn't return results
                        # We just know the function was called
                        function_call_info["note"] = "Executed automatically by SDK"
                        function_calls_made.append(function_call_info)
        
        # Fallback to response.text if no parts found
        if not response_text:
            response_text = response.text or ""
        
        # Clean JSON wrapper if present (Gemini sometimes wraps responses in ```json blocks)
        if response_text.strip().startswith('```json'):
            response_text = response_text.strip()[7:-3].strip()
        
        # Build response structure
        result = {
            "content": [
                {"thinking": thinking_text},
                {"text": response_text}
            ]
        }
        
        # Add function calls if any were made
        if function_calls_made:
            result["function_calls"] = function_calls_made
        
        return json.dumps(result, indent=2)
    
    def _extract_thinking_and_content(self, response):
        """Extract thinking and content from response following Google's simple pattern
        
        Uses direct attribute access: response.candidates[0].content.parts[0]
        as shown in Google's documentation examples
        """
        thinking_text = ""
        response_text = ""
        function_calls = []
        
        # Follow Google's direct access pattern
        # Fix: Add None check for content - Gemini API can return None content in some cases
        # if response.candidates and response.candidates[0].content.parts:  # Old - crashes when content is None
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'thought') and part.thought and self.enable_thinking:
                    thinking_text += f"{part.text or ''}\n"
                elif hasattr(part, 'text') and part.text:
                    response_text += part.text
                elif hasattr(part, 'function_call'):
                    function_calls.append(part.function_call)
        
        return thinking_text, response_text, function_calls

    # BACKUP: Original sequential version (commented out)
    # def _execute_all_function_calls(self, function_calls):
    #     """Execute all function calls and return results"""
    #     function_calls_made = []
    #     web_results = []
    #     function_responses = []
    #     
    #     for function_call in function_calls:
    #         # Execute the function call
    #         function_call_result = self.execute_function_call(function_call)
    #         
    #         # Track function call details
    #         function_call_info = {
    #             "name": function_call.name,
    #             "args": dict(function_call.args) if hasattr(function_call, 'args') else {},
    #             "result": function_call_result,
    #             "success": function_call_result.get("success", False),
    #             "error": function_call_result.get("error") if not function_call_result.get("success", False) else None
    #         }
    #         function_calls_made.append(function_call_info)
    #         
    #         # Format web search results
    #         if function_call.name == "search_web" and function_call_result.get("success"):
    #             search_data = function_call_result.get("data", [])
    #             query = function_call_result.get("query", "")
    #             
    #             for item in search_data:
    #                 web_results.append({
    #                     "source": "web_search_via_mcp",
    #                     "title": item.get("title", ""),
    #                     "content": item.get("snippet", ""),
    #                     "url": item.get("link", ""),
    #                     "display_link": item.get("displayLink", ""),
    #                     "search_query": query,
    #                     "timestamp": datetime.now().isoformat()
    #                 })
    #         
    #         # Create function response part for conversation
    #         function_response = types.Part.from_function_response(
    #             name=function_call.name,
    #             response=function_call_result
    #         )
    #         function_responses.append(function_response)
    #     
    #     return function_calls_made, web_results, function_responses
    
    # NEW: Parallel version using asyncio.gather
    def _execute_all_function_calls(self, function_calls):
        """Execute all function calls in PARALLEL and return results

        Uses ContextThreadPoolExecutor for concurrent execution of registered tool functions.
        This preserves LangGraph context and ensures parallel functions execute simultaneously.
        Matches the parallel execution pattern used in workflow.py:parallel_executor_node
        """
        from concurrent.futures import as_completed

        if len(function_calls) > 1:
            logger.info(f"[Function Execution] Executing {len(function_calls)} functions in PARALLEL")

        function_calls_made = []
        function_responses = []

        # Execute all functions in parallel using LangGraph-compatible thread pool
        with ContextThreadPoolExecutor(max_workers=len(function_calls)) as executor:
            # Submit all function calls
            future_to_fc = {}

            for fc in function_calls:
                # Find registered function by name in self.tools
                func = next((f for f in self.tools if f.__name__ == fc.name), None)

                if func:
                    args = dict(fc.args) if hasattr(fc, 'args') else {}
                    future = executor.submit(func, **args)
                    future_to_fc[future] = fc
                else:
                    # Function not found - create error entry
                    future_to_fc[None] = fc

            # Collect results in original order to match function_calls sequence
            for fc in function_calls:
                matching_future = next((f for f, call in future_to_fc.items() if call == fc), None)

                if matching_future is None:
                    result = {"success": False, "error": f"Unknown function: {fc.name}"}
                else:
                    try:
                        result = matching_future.result()
                    except Exception as e:
                        result = {"success": False, "error": str(e)}

                # Track function call metadata (exclude binary/multimodal fields for JSON serialization)
                result_for_metadata = {k: v for k, v in result.items()
                                      if k not in ["image_bytes", "mime_type", "file_uri"]}

                function_call_info = {
                    "name": fc.name,
                    "args": dict(fc.args) if hasattr(fc, 'args') else {},
                    "result": result_for_metadata,  # Exclude image_bytes
                    "success": result.get("success", False),
                    "error": result.get("error") if not result.get("success", False) else None
                }
                function_calls_made.append(function_call_info)

                # Create function response part(s) for conversation
                # Use multimodal builder (returns list of Parts)
                parts = self._build_multimodal_function_response(fc, result)
                function_responses.extend(parts)

        return function_calls_made, function_responses

    def _build_multimodal_function_response(self, function_call, result):
        """Build function response with optional multimodal content

        Follows Google's nested parts structure for multimodal function responses.
        Uses filename-based $ref system for clear image identification.

        Args:
            function_call: The function call object
            result: Function return dict with optional multimodal_response

        Returns:
            List containing single Part with nested multimodal data (if present)
        """
        # Extract response data without multimodal_response container
        response_data = {k: v for k, v in result.items() if k != "multimodal_response"}

        # Check if multimodal response is enabled
        if not self.gemini_configs.get('enable_multimodal_responses'):
            return [types.Part.from_function_response(
                name=function_call.name,
                response=response_data
            )]

        # Get multimodal_response container
        multimodal_resp = result.get("multimodal_response")
        if not multimodal_resp:
            return [types.Part.from_function_response(
                name=function_call.name,
                response=response_data
            )]

        parts = []

        # Process generated images (always list)
        for gen in multimodal_resp.get("generated", []):
            filename = gen["file_uri"].split('/')[-1]

            # Add $ref structure to response_data
            response_data[filename] = {
                "$ref": filename,
                "description": gen["description"]
            }

            # Add multimodal part
            parts.append(types.FunctionResponsePart(
                file_data=types.FunctionResponseFileData(
                    mime_type=gen["mime_type"],
                    display_name=filename,
                    file_uri=gen["file_uri"]
                )
            ))

        # Process reference images (always list)
        for ref in multimodal_resp.get("references", []):
            filename = ref["file_uri"].split('/')[-1]

            # Add $ref structure to response_data
            response_data[filename] = {
                "$ref": filename,
                "description": ref["description"]
            }

            # Add multimodal part
            parts.append(types.FunctionResponsePart(
                file_data=types.FunctionResponseFileData(
                    mime_type=ref["mime_type"],
                    display_name=filename,
                    file_uri=ref["file_uri"]
                )
            ))

        logger.info(f"[Handler] Built function response with {len(parts)} multimodal parts ({len(multimodal_resp.get('generated', []))} generated + {len(multimodal_resp.get('references', []))} references)")

        # Return function_response with nested multimodal parts
        return [types.Part.from_function_response(
            name=function_call.name,
            response=response_data,
            parts=parts
        )]

    # COMMENTED OUT: Orphaned method - replaced by _handle_function_calling_from_stream
    # References removed self.max_function_calling_rounds parameter
    # Keeping for reference but not used anywhere in codebase
    # def _handle_function_calling_response(self, response, client, contents, config):
    #     """Handle iterative function calling workflow with parallel execution support
    #
    #     Supports both parallel function calls within rounds and sequential rounds
    #     for complex workflows like: search → analyze → fetch content → synthesize
    #     """
    #     # Extract initial thinking and function calls
    #     thinking_text, response_text, function_calls = self._extract_thinking_and_content(response)
    #     thinking_text = f"[Round 1 thinking] {thinking_text}" if thinking_text else ""
    #
    #     all_function_calls_made = []
    #     all_web_results = []
    #     current_response = response
    #
    #     # Iterative function calling loop with configurable max rounds
    #     for round_num in range(1, self.max_function_calling_rounds + 1):
    #
    #         if not function_calls:
    #             # No more function calls needed, we have our final response
    #             break
    #
    #         # Add current response to conversation (preserves thinking)
    #         contents.append(current_response.candidates[0].content)
    #
    #         # Execute all function calls in parallel for this round
    #         function_calls_made, web_results, function_responses = self._execute_all_function_calls(function_calls)
    #
    #         # Accumulate results across all rounds
    #         all_function_calls_made.extend(function_calls_made)
    #         all_web_results.extend(web_results)
    #
    #         # Add function responses to conversation
    #         if function_responses:
    #             contents.append(types.Content(role="user", parts=function_responses))
    #
    #         # Check if this is the last round
    #         if round_num == self.max_function_calling_rounds:
    #             # Force final response - remove tools from config to disable function calling
    #             final_config_dict = {k: v for k, v in config.__dict__.items() if k not in ['tools', 'tool_config']}
    #             final_config = types.GenerateContentConfig(**final_config_dict)
    #             thinking_text += f"\n[Final round {round_num}] Forcing final response (max rounds reached)"
    #
    #             # Add final round summarization instructions to contents
    #             final_instructions = types.Part.from_text(
    #                 text="IMPORTANT: This is your final response. Summarize and distill the key facts from ALL search results and fetched content. Present only the factual information found. Be concise and stick to the facts - no recommendations, analysis, or expansion."
    #             )
    #             contents.append(types.Content(role="user", parts=[final_instructions]))
    #
    #             try:
    #                 final_response = client.models.generate_content(
    #                     model=self.model_name,
    #                     contents=contents,
    #                     config=final_config
    #                 )
    #
    #                 final_thinking, final_text, _ = self._extract_thinking_and_content(final_response)
    #                 if final_thinking:
    #                     thinking_text += f"\n[Final synthesis] {final_thinking}"
    #                 response_text = final_text or (final_response.text or "Final response not generated")
    #
    #             except Exception as e:
    #                 thinking_text += f"\n[Error in final round] {str(e)}"
    #                 response_text += f" [Max rounds reached, error getting final response: {str(e)}]"
    #
    #             break
    #
    #         else:
    #             # Not the last round - get next response and check for more function calls
    #             try:
    #                 next_response = client.models.generate_content(
    #                     model=self.model_name,
    #                     contents=contents,
    #                     config=config
    #                 )
    #
    #                 # Extract thinking, response text, and check for more function calls
    #                 next_thinking, next_text, next_function_calls = self._extract_thinking_and_content(next_response)
    #
    #                 if next_thinking:
    #                     thinking_text += f"\n[Round {round_num + 1} thinking] {next_thinking}"
    #
    #                 # Update for next iteration
    #                 current_response = next_response
    #                 function_calls = next_function_calls
    #
    #                 # If no more function calls, this is our final response
    #                 if not next_function_calls:
    #                     response_text = next_text
    #                     thinking_text += f"\n[Round {round_num + 1}] No more function calls needed, task complete"
    #                     break
    #
    #             except Exception as e:
    #                 thinking_text += f"\n[Error in round {round_num + 1}] {str(e)}"
    #                 response_text += f" [Function calls completed but error in round {round_num + 1}: {str(e)}]"
    #                 break
    #
    #     # Return JSON structure
    #     result = {
    #         "content": [
    #             {"thinking": thinking_text.strip()},
    #             {"text": response_text or "No response generated"}
    #         ]
    #     }
    #
    #     # Add metadata for debugging
    #     if all_function_calls_made:
    #         result["function_calls"] = all_function_calls_made
    #         if all_web_results:
    #             result["web_results"] = all_web_results
    #
    #     return json.dumps(result, indent=2)

    def _handle_function_calling_from_stream(self, initial_function_calls, initial_thinking,
                                            initial_text, accumulated_parts, client, contents, config,
                                            stream_callback=None):
        """Manual function calling after streaming completes

        Called when streaming detected function calls - enters manual execution loop
        with accumulated calls from the stream. Uses current parameter system.

        IMPORTANT: Preserves thought signatures by using accumulated Parts from ALL streaming
        chunks. For parallel function calls, only the first function_call Part has a signature.
        We must preserve the EXACT order and structure as received from streaming.

        Args:
            initial_function_calls: Function calls detected during streaming
            initial_thinking: Accumulated thinking text from stream
            initial_text: Accumulated response text from stream
            accumulated_parts: ALL Parts from ALL streaming chunks (preserves thought signatures)
            client: Gemini client instance
            contents: Conversation history so far
            config: GenerateContentConfig for subsequent calls

        Returns:
            JSON string with thinking, text, and function call metadata
        """
        # Use maximum_remote_calls (current param, not old max_function_calling_rounds)
        max_rounds = self.gemini_configs.get('maximum_remote_calls', 20)

        logger.info(f"[Manual Function Calling] Starting manual execution with {len(initial_function_calls)} function call(s), max {max_rounds} rounds")

        thinking_text = f"[Round 1 thinking] {initial_thinking}" if initial_thinking else ""
        response_text = initial_text
        function_calls = initial_function_calls

        all_function_calls_made = []

        # CRITICAL: Must use accumulated Parts to preserve thought signatures
        # Per Gemini docs: https://ai.google.dev/gemini-api/docs/thought-signatures
        # "The first functionCall part in each step MUST include its thought_signature"
        # Reconstructing Parts manually loses these signatures - fail fast if this happens
        if not accumulated_parts:
            error_msg = (
                f"[Manual Function Calling] CRITICAL: accumulated_parts is empty but "
                f"{len(initial_function_calls)} function call(s) detected. "
                f"Cannot proceed without thought signatures. "
                f"See: https://ai.google.dev/gemini-api/docs/thought-signatures"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        contents.append(types.Content(role="model", parts=accumulated_parts))

        # Multi-round loop
        for round_num in range(1, max_rounds + 1):

            if not function_calls:
                break

            # Execute function calls in parallel
            function_calls_made, function_responses = self._execute_all_function_calls(function_calls)
            all_function_calls_made.extend(function_calls_made)

            logger.info(f"[Manual Function Calling] Round {round_num}: Executed {len(function_calls_made)} function(s)")

            # Add function responses to conversation (role="tool" for Gemini 3)
            if function_responses:
                contents.append(types.Content(role="tool", parts=function_responses))

            # Check if this is the last round
            if round_num == max_rounds:
                logger.warning(f"[Manual Function Calling] Max rounds ({max_rounds}) reached, forcing final response")
                # Force final response - remove tools from config
                final_config_dict = {k: v for k, v in config.__dict__.items()
                                   if k not in ['tools', 'tool_config']}
                final_config = types.GenerateContentConfig(**final_config_dict)
                thinking_text += f"\n[Final round {round_num}] Forcing final response (max rounds reached)"

                final_instructions = types.Part.from_text(
                    text="IMPORTANT: This is your final response. Summarize and distill the key facts from ALL function results. Present only the factual information found. Be concise and stick to the facts."
                )
                contents.append(types.Content(role="user", parts=[final_instructions]))

                try:
                    # Stream final response to capture thinking
                    final_thinking = ""
                    final_text = ""
                    final_accumulated_parts = []

                    for chunk in client.models.generate_content_stream(
                        model=self.model_name,
                        contents=contents,
                        config=final_config
                    ):
                        # Accumulate ALL parts to preserve thought signatures
                        if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                            for part in chunk.candidates[0].content.parts:
                                final_accumulated_parts.append(part)

                        # Accumulate text
                        if hasattr(chunk, 'text') and chunk.text:
                            final_text += chunk.text
                            if stream_callback:
                                stream_callback('token', chunk.text)

                        # Capture thinking from parts
                        if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                            for part in chunk.candidates[0].content.parts:
                                if hasattr(part, 'thought') and part.thought:
                                    if hasattr(part, 'text') and part.text:
                                        final_thinking += part.text
                                        if stream_callback:
                                            stream_callback('thinking', part.text)

                    if final_thinking:
                        thinking_text += f"\n[Final synthesis] {final_thinking}"
                    response_text = final_text or "Final response not generated"

                except Exception as e:
                    logger.error(f"[Manual Function Calling] Error in final round: {str(e)}")
                    thinking_text += f"\n[Error in final round] {str(e)}"
                    response_text += f" [Max rounds reached, error getting final response: {str(e)}]"

                break

            else:
                # Not the last round - get next response using STREAMING (like Round 1)
                try:
                    # Stream the response to capture thinking from all chunks
                    next_thinking = ""
                    next_text = ""
                    next_function_calls = []
                    next_accumulated_parts = []

                    for chunk in client.models.generate_content_stream(
                        model=self.model_name,
                        contents=contents,
                        config=config
                    ):
                        # Accumulate ALL parts and check for thinking/function calls in single iteration
                        if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                            for part in chunk.candidates[0].content.parts:
                                # Accumulate part to preserve thought signatures
                                next_accumulated_parts.append(part)

                                # Check for thinking
                                if hasattr(part, 'thought') and part.thought:
                                    if hasattr(part, 'text') and part.text:
                                        next_thinking += part.text
                                        if stream_callback:
                                            stream_callback('thinking', part.text)

                                # Check for function calls
                                if hasattr(part, 'function_call') and part.function_call is not None:
                                    next_function_calls.append(part.function_call)

                        # Accumulate text from chunks
                        if hasattr(chunk, 'text') and chunk.text:
                            next_text += chunk.text
                            if stream_callback:
                                stream_callback('token', chunk.text)

                    # Add captured thinking to overall thinking
                    if next_thinking:
                        thinking_text += f"\n[Round {round_num + 1} thinking] {next_thinking}"

                    # Update for next iteration - construct response object from last chunk
                    current_response = chunk  # Last chunk has complete candidates
                    function_calls = next_function_calls

                    # If no more function calls, this is our final response
                    if not next_function_calls:
                        response_text = next_text
                        thinking_text += f"\n[Round {round_num + 1}] No more function calls needed, task complete"
                        logger.info(f"[Manual Function Calling] Completed after {round_num + 1} rounds with {len(all_function_calls_made)} total function call(s)")
                        break
                    else:
                        # CRITICAL: Append model's response to preserve thought signatures for next round
                        # Per Google's docs: thought signatures must be preserved across multi-turn function calling
                        # Construct Content from accumulated parts
                        next_content = types.Content(role="model", parts=next_accumulated_parts)
                        contents.append(next_content)
                        logger.info(f"[Manual Function Calling] Round {round_num + 1}: Model requested {len(next_function_calls)} more function call(s)")

                except Exception as e:
                    logger.error(f"[Manual Function Calling] Error in round {round_num + 1}: {str(e)}")
                    thinking_text += f"\n[Error in round {round_num + 1}] {str(e)}"
                    response_text += f" [Function calls completed but error in round {round_num + 1}: {str(e)}]"
                    break

        # Return JSON structure
        result = {
            "content": [
                {"thinking": thinking_text.strip()},
                {"text": response_text or "No response generated"}
            ]
        }

        # Add metadata for debugging
        if all_function_calls_made:
            result["function_calls"] = all_function_calls_made

        logger.info(f"[Manual Function Calling] Returning response with {len(all_function_calls_made)} function call(s) executed")

        return json.dumps(result, indent=2)

    def _call(self, prompt, stop=None, run_manager=None, images=None, documents=None, add_context=True, context_content="", response_schema=None, **kwargs) -> str:
        """Call Gemini with prompt, optional images, documents, thinking, context injection, and auto-detected structured output - now with streaming"""
        try:
            client = self.setup_gemini()

            # Get streaming callback if provided
            stream_callback = kwargs.get('stream_callback')

            # Prompt should be a string
            prompt_text = prompt

            # Inject context if enabled and available
            final_prompt = prompt_text
            if add_context and context_content:
                final_prompt = f"{prompt_text}\n\n### Current Context:\n{context_content}"

            # Capture prompt for debugging with real token counts
            state = kwargs.get('state')
            if state:
                self._capture_prompt_debug(final_prompt, context_content, state, template=prompt_text, system_instruction=self.system_instruction)

            # Token counting debug (drop-in, zero overhead if disabled)
            from .config import DEBUG_TOKEN_COUNTING
            if DEBUG_TOKEN_COUNTING:
                try:
                    # Count final prompt only (most important)
                    final_tokens = self.count_tokens(final_prompt, client)
                    model_limit = 1_048_576  # Gemini 2.5 Pro input limit

                    # Simple log
                    agent_name = kwargs.get('agent_name', 'unknown')
                    print(f"[TOKEN] {agent_name}: {final_tokens:,} tokens ({final_tokens*100/model_limit:.1f}% of {model_limit:,})")
                except Exception as e:
                    print(f"[TOKEN DEBUG] Error counting tokens for {kwargs.get('agent_name', 'unknown')}: {e}")

            # Build content list
            contents = []

            # Add documents first if provided (convert universal format to Gemini Parts)
            if documents:
                for doc in documents:
                    # Universal format: {"data": bytes, "media_type": str}
                    part = types.Part.from_bytes(
                        data=doc['data'],
                        mime_type=doc['media_type']
                    )
                    contents.append(part)

            # Add images if provided and enabled
            if self.enable_images and images:
                for image in images:
                    if 'file_uri' in image:
                        # Direct GCS URI - zero latency (Google fetches internally)
                        contents.append(types.Part.from_uri(
                            file_uri=image['file_uri'],
                            mime_type=image.get('media_type', 'image/png')
                        ))
                    elif 'data' in image and 'media_type' in image:
                        # Base64 format
                        import base64
                        image_data = base64.b64decode(image['data'])
                        contents.append(types.Part.from_bytes(
                            data=image_data,
                            mime_type=image['media_type']
                        ))

            # Add text prompt as Part (always last)
            contents.append(types.Part.from_text(text=final_prompt))
            
            # Check if structured output is needed via auto-detected schema
            use_structured_output = response_schema is not None
            schema_to_use = None

            if use_structured_output:
                try:
                    # Load Gemini-specific schema
                    schema_to_use = get_schema(response_schema, "gemini")
                except (ValueError, ImportError, AttributeError) as e:
                    print(f"[GeminiLLM] Error loading schema '{response_schema}': {e}")
                    use_structured_output = False
            
            # Prepare generation config
            config_params = {
                "temperature": self.gemini_configs['temperature'],
                # top_p=0.95,
                "max_output_tokens": self.gemini_configs['max_output_tokens'],
                "safety_settings": [
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE"
                    )
                ],
                "thinking_config": (
                    types.ThinkingConfig(
                        thinking_level=self.thinking_level,
                        include_thoughts=True
                    ) if "gemini-3" in self.model_name.lower() else
                    types.ThinkingConfig(
                        thinking_budget=self.thinking_budget_tokens,
                        include_thoughts=True
                    )
                ) if self.enable_thinking else None,
            }

            # Use cached content if available, otherwise use system instruction
            if self.cached_content_name:
                # When using cache: only pass cached_content (system_instruction/tools already in cache)
                config_params["cached_content"] = self.cached_content_name
                logger.info(f"[Gemini LLM] Using cached content: {self.cached_content_name}")
            else:
                # When NOT using cache: pass system_instruction + tools
                if self.system_instruction:
                    config_params["system_instruction"] = self.system_instruction

                # Add function calling configuration
                # Support both function declarations and direct Python functions
                if self.tools:
                    # Direct Python functions for automatic calling
                    config_params["tools"] = self.tools

                    # Configure automatic function calling with maximum_remote_calls
                    if self.gemini_configs.get('automatic_function_calling'):
                        max_calls = self.gemini_configs.get('maximum_remote_calls', 20)
                        config_params["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                            maximum_remote_calls=max_calls
                        )
                    elif self.gemini_configs.get('automatic_function_calling') is False:
                        # CORRECT: Disable SDK automatic execution
                        # SDK will return function calls in response but NOT execute them
                        # We'll execute manually in our handler
                        config_params["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                            disable=True
                        )
                elif self.function_declarations:
                    # Manual function declarations (old pattern)
                    tools = types.Tool(function_declarations=self.function_declarations)
                    config_params["tools"] = [tools]

                # Add tool configuration if provided (but don't override manual mode config)
                if self.tool_config and self.gemini_configs.get('automatic_function_calling') is not False:
                    config_params["tool_config"] = self.tool_config

            # Add structured output configuration if needed
            if use_structured_output and schema_to_use:
                config_params["response_mime_type"] = "application/json"
                config_params["response_schema"] = schema_to_use

            config = types.GenerateContentConfig(**config_params)
            
            # Stream generation and accumulate response
            thinking_text = ""
            response_text = ""
            function_calls = []
            accumulated_parts = []  # Accumulate ALL parts from all chunks to preserve thought signatures

            # DEBUG: Enable debugging
            debug_streaming = False  # Set to True to debug
            chunk_count = 0

            for chunk in client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=config
            ):
                chunk_count += 1

                # CRITICAL: Accumulate ALL parts from ALL chunks to preserve thought signatures
                # Parallel function calls have signature only on first call, which might be in any chunk
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        accumulated_parts.append(part)
                
                # DEBUG: Log raw chunk data
                if debug_streaming:
                    print(f"\n[DEBUG Chunk {chunk_count}]")
                    print(f"  chunk.text = {repr(chunk.text if hasattr(chunk, 'text') else 'NO ATTR')}")
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        # Fix: Add None check for content - Gemini API can return chunks with content=None
                        # if chunk.candidates[0].content.parts:  # Old - crashes when content is None
                        if chunk.candidates[0].content and chunk.candidates[0].content.parts:
                            for i, part in enumerate(chunk.candidates[0].content.parts):
                                print(f"  part[{i}].text = {repr(part.text if hasattr(part, 'text') else 'NO ATTR')}")
                                print(f"  part[{i}].thought = {part.thought if hasattr(part, 'thought') else 'NO ATTR'}")
                
                # Use chunk.text for simple text accumulation (SDK recommended approach)
                if hasattr(chunk, 'text') and chunk.text:
                    response_text += chunk.text
                    if stream_callback:
                        stream_callback('token', chunk.text)
                
                # Also check parts for thinking and function calls
                # Fix: Add None check for content - Gemini API can return chunks with content=None during function calling
                # if chunk.candidates and chunk.candidates[0].content.parts:  # Old - crashes when content is None
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        # Check for thinking text specifically
                        if hasattr(part, 'thought') and part.thought:
                            # This part contains thinking - accumulate it
                            if hasattr(part, 'text') and part.text:
                                thinking_text += part.text
                                if stream_callback:
                                    stream_callback('thinking', part.text)
                        
                        # Check for function calls
                        if hasattr(part, 'function_call') and part.function_call is not None:
                            # Store function call for potential execution
                            function_calls.append(part.function_call)
                            # Note: Function calls are not streamed to UI
            
            # DEBUG: Log accumulated results
            if debug_streaming:
                print(f"\n[DEBUG Summary]")
                print(f"  Total chunks: {chunk_count}")
                print(f"  Thinking accumulated: {len(thinking_text)} chars")
                print(f"  Text accumulated: {len(response_text)} chars")
                print(f"  Function calls: {len(function_calls)}")
            
            # If function calls were made and tools are configured, handle them
            if function_calls and self.tools:
                # Check if manual mode requested
                if self.gemini_configs.get('automatic_function_calling') is False:
                    logger.info(f"[Function Calling] Manual mode triggered with {len(function_calls)} function call(s)")
                    # Stream-then-Manual: Enter manual execution loop
                    # Convert contents (list of Parts) to proper Content list for multi-turn
                    content_list = [types.Content(role="user", parts=contents)]

                    return self._handle_function_calling_from_stream(
                        initial_function_calls=function_calls,
                        initial_thinking=thinking_text,
                        initial_text=response_text,
                        accumulated_parts=accumulated_parts,  # All Parts from all chunks (preserves signatures)
                        client=client,
                        contents=content_list,  # Proper Content list for multi-turn
                        config=config,
                        stream_callback=stream_callback
                    )
                else:
                    # AFC mode: SDK already executed functions and handled signatures.
                    # Do NOT run manual loop - this causes thought_signature errors.
                    # Just return what we streamed.
                    logger.info(f"[Automatic Function Calling] {len(function_calls)} function call(s) "
                                f"detected. SDK handles execution. Returning streamed response.")

                    if self.enable_thinking:
                        return json.dumps({
                            "content": [
                                {"thinking": thinking_text},
                                {"text": response_text},
                            ]
                        })
                    return response_text
            
            # Return in expected JSON format
            return json.dumps({
                "content": [
                    {"thinking": thinking_text},
                    {"text": response_text}
                ]
            })
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc().replace('\n', ' | ')
            print(f"[GEMINI LLM ERROR] {type(e).__name__}: {str(e)} | TRACEBACK: {tb}")
            # Return structured payload matching expected schema
            return json.dumps({
                "content": [
                    {"thinking": ""},
                    {"text": f"LLM error: {str(e)}"}
                ]
            })
    
    async def astream(
        self,
        prompt: str,
        stream_callback=None,
        add_context: bool = True,
        **kwargs
    ):
        """
        Async streaming generation for real-time token-by-token output.
        Uses Gemini's async streaming API for WebSocket integration.
        
        Args:
            prompt: Input prompt
            stream_callback: Optional async callback for each chunk
            add_context: Whether to add context
            
        Returns:
            Complete generated text
        """
        try:
            # Initialize client with proper credentials and location
            client = self.setup_gemini()

            # Build contents (simple text-only for streaming)
            contents = [types.Part.from_text(text=prompt)]
            
            # Prepare configuration
            config_dict = self.gemini_configs.copy()
            config_dict.update(kwargs)
            
            # Add JSON mode if needed
            if hasattr(self, 'json_mode') and self.json_mode:
                config_dict['response_mime_type'] = 'application/json'
            
            # Configure for streaming
            config = genai.GenerateContentConfig(**config_dict)
            
            # Stream generation
            full_response = []
            thinking_text = ""
            in_thinking = False
            
            async for chunk in client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=config
            ):
                if chunk.text:
                    # Handle thinking tags if enabled
                    if self.enable_thinking and '<thinking>' in chunk.text:
                        in_thinking = True
                        thinking_text += chunk.text
                    elif self.enable_thinking and '</thinking>' in chunk.text:
                        in_thinking = False
                        thinking_text += chunk.text
                    elif in_thinking:
                        thinking_text += chunk.text
                    else:
                        # Regular content - stream it
                        full_response.append(chunk.text)
                        if stream_callback:
                            await stream_callback(chunk.text)
            
            # Return complete response
            complete_text = "".join(full_response)
            
            # If thinking was captured, structure the response
            if self.enable_thinking and thinking_text:
                return json.dumps({
                    "content": [
                        {"thinking": thinking_text},
                        {"text": complete_text}
                    ]
                })
            
            return complete_text
            
        except Exception as e:
            error_msg = f"Gemini async streaming error: {e}"
            print(error_msg)
            if stream_callback:
                await stream_callback(f"\n[Error: {str(e)}]")
            return json.dumps({
                "content": [
                    {"thinking": ""},
                    {"text": error_msg}
                ]
            })

    def _capture_prompt_debug(self, prompt, context, state, template=None, system_instruction=None):
        """Capture full prompt for debugging with real Gemini token counts"""
        from .config import DEBUG_CONTEXT

        if not DEBUG_CONTEXT or not state:
            return

        agent_name = state.get("current_agent", "unknown")

        if "debug_prompts" not in state:
            state["debug_prompts"] = {}

        turn_number = state.get("turn_number", 0)
        turn_key = f"turn_{turn_number}"

        if turn_key not in state["debug_prompts"]:
            state["debug_prompts"][turn_key] = {}

        # Calculate real token counts using Gemini API (free)
        system_instruction_tokens = 0
        template_tokens = 0
        context_tokens = 0
        total_tokens = 0

        try:
            if system_instruction:
                system_instruction_tokens = self.count_tokens(system_instruction)
            if template:
                template_tokens = self.count_tokens(template)
            if context:
                context_tokens = self.count_tokens(context)
            prompt_tokens = self.count_tokens(prompt)
            total_tokens = system_instruction_tokens + prompt_tokens
        except Exception as e:
            # Fallback to estimation if count error
            print(f"[Token Count] Error for {agent_name}, using estimation: {e}")
            system_instruction_tokens = len(system_instruction) // 4 if system_instruction else 0
            template_tokens = len(template) // 4 if template else 0
            context_tokens = len(context) // 4 if context else 0
            prompt_tokens = len(prompt) // 4
            total_tokens = system_instruction_tokens + prompt_tokens

        prompt_data = {
            "system_instruction": system_instruction or "",
            "system_instruction_token_count": system_instruction_tokens,
            "system_instruction_length_chars": len(system_instruction) if system_instruction else 0,
            "context_section": prompt,  # After Phase 1, prompt parameter contains context_content
            "full_prompt": f"{system_instruction}\n\n{prompt}" if system_instruction else prompt,  # Reconstructed full
            "timestamp": datetime.now().isoformat(),
            "token_count": total_tokens,
            "template_token_count": template_tokens,
            "context_token_count": prompt_tokens,  # Uses official API count
            "prompt_length_chars": len(prompt),
            "template_length_chars": len(template) if template else 0,
            "context_length_chars": len(prompt),  # Context is in prompt parameter now
            "cache_used": bool(self.cached_content_name),
            "cache_name": self.cached_content_name if self.cached_content_name else None
        }

        # Check if agent already has data - if so, convert to array for sub-agents
        if agent_name in state["debug_prompts"][turn_key]:
            existing = state["debug_prompts"][turn_key][agent_name]
            # Convert to array if not already
            if not isinstance(existing, list):
                state["debug_prompts"][turn_key][agent_name] = [existing]
            # Append new prompt
            state["debug_prompts"][turn_key][agent_name].append(prompt_data)
        else:
            # First call from this agent
            state["debug_prompts"][turn_key][agent_name] = prompt_data

    @property
    def _llm_type(self):
        return "gemini_vertex"

def get_llm(**kwargs):
    """Get Gemini LLM instance

    Args:
        **kwargs: Configuration parameters passed to GeminiVertexLLM

    Returns:
        GeminiVertexLLM instance
    """
    # Handle tools parameter from gemini_configs
    if 'gemini_configs' in kwargs and 'tools' in kwargs['gemini_configs']:
        tools = kwargs['gemini_configs'].pop('tools')
        kwargs['tools'] = tools

    # Remove any model parameter (always use Gemini)
    kwargs.pop('model', None)

    # Remove unused config parameters
    kwargs.pop('fallback_configs', None)
    kwargs.pop('extra_configs', None)

    return GeminiVertexLLM(**kwargs)



# Schemas moved to src/schemas/ directory
# Use get_schema(agent_name, model_type) to load schemas dynamically

