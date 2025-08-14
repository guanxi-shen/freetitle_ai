"""
Tool-based image generation system

Tools:
- nano_banana: Gemini 3 Pro Image generation with reasoning capabilities

Each tool specializes in different use cases:
- nano_banana: High-quality image generation, character consistency, complex prompts
"""

import os
import json
import base64
from pathlib import Path
from io import BytesIO
from typing import Optional, Dict, Any, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image as PILImage
from google import genai
from google.genai import types
from ...core.config import (
    PROJECT_ID, CREDENTIALS, LOCATION, GEMINI_ASPECT_RATIOS
)


# ==============================================================================
# ASYNC TASK INFRASTRUCTURE
# ==============================================================================

# Module-level task registry for async storyboard generation
# Async infrastructure removed - now using synchronous blocking pattern


def _infer_asset_type_from_task_id(task_id: str) -> str:
    """
    Infer GCS asset_type from task_id prefix.

    Task ID naming convention:
    - storyboard_* → "storyboards"
    - supplementary_* → "supplementary"
    - Future agents can add their prefixes here

    Args:
        task_id: Task identifier with prefix

    Returns:
        GCS folder path (e.g., "storyboards", "supplementary")
    """
    if task_id.startswith("storyboard_"):
        return "storyboards"
    elif task_id.startswith("supplementary_"):
        return "supplementary"
    else:
        # Fallback for legacy or unknown task IDs
        print(f"[Asset Type Inference] Unknown task_id prefix: {task_id}, defaulting to 'storyboards'")
        return "storyboards"


def convert_to_gs_uri(url: str) -> str:
    """
    Convert any GCS URL format to gs:// URI

    Args:
        url: GCS URL in any format (gs://, https://storage.googleapis.com, signed URL)

    Returns:
        Clean gs:// URI without query parameters
    """
    if url.startswith("gs://"):
        return url.split("?")[0]
    elif url.startswith("https://storage.googleapis.com"):
        parts = url.replace("https://storage.googleapis.com/", "").split("?")[0].split("/", 1)
        return f"gs://{parts[0]}/{parts[1]}"
    return url


def detect_mime_type(uri: str) -> str:
    """
    Detect MIME type from file extension

    Args:
        uri: File URI or path

    Returns:
        MIME type string (defaults to image/png)
    """
    ext = Path(uri).suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.gif': 'image/gif'
    }
    return mime_types.get(ext, 'image/png')


def nano_banana(
    prompt: str,
    reference_images: Optional[List[Union[str, Dict[str, Any]]]] = None,
    aspect_ratio: str = "vertical",
    session_id: str = None,
    save_path: Optional[str] = None,
    reference_metadata: Optional[Dict[str, Any]] = None,
    check_aspect_ratio: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Nano Banana: Gemini 3 Pro Image generation with reasoning capabilities.
    Best for: High-quality image generation, character consistency, complex prompts with reasoning.

    Args:
        prompt: Text description for image generation
        reference_images: Optional list of reference images (max 14)
        aspect_ratio: Aspect ratio preset (vertical, horizontal, square, etc.)
        session_id: Session ID for tracking
        save_path: Optional path to save generated image (deprecated)
        reference_metadata: Optional metadata about references
        check_aspect_ratio: Validate aspect ratio match
        **kwargs: Additional tool-specific parameters

    Returns:
        Dictionary containing:
        - success: bool indicating if generation succeeded
        - image_bytes: Raw image bytes
        - image_data: Base64 encoded image data
        - gcs_url: GCS URL if uploaded
        - error: Error message if failed
    """
    try:
        # Validate reference image count (upgraded limit: 14 total, up to 6 objects + 5 humans)
        if reference_images and len(reference_images) > 14:
            error_msg = (f"Image generation error: Too many reference images provided. "
                        f"Maximum is 14, but received {len(reference_images)}. "
                        f"Please reduce the number of reference images.")
            return {
                "success": False,
                "error": error_msg,
                "image_data": None,
                "text_response": None,
                "saved_path": None
            }

        # Gemini 3 Pro Image generation
        print(f"[Image Tools] Using Gemini 3 Pro Image")

        # Initialize Gemini client with credentials - MUST use global for image generation
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location="global",  # Image generation requires global location
            credentials=CREDENTIALS
        )
        
        # Build content list
        contents = []

        # Add reference images if provided
        if reference_images:
            for ref_image in reference_images:
                if isinstance(ref_image, str):
                    # GCS URL - use direct URI reference (zero latency)
                    if ref_image.startswith(("gs://", "https://storage.googleapis.com")):
                        gs_uri = convert_to_gs_uri(ref_image)
                        mime_type = detect_mime_type(gs_uri)
                        contents.append(types.Part.from_uri(
                            file_uri=gs_uri,
                            mime_type=mime_type
                        ))
                        print(f"[Image Tools] Using GCS reference: {gs_uri}")
                    # Local file path - load and upload as bytes
                    elif os.path.exists(ref_image):
                        with open(ref_image, 'rb') as f:
                            image_bytes = f.read()
                            mime_type = detect_mime_type(ref_image)
                            contents.append(types.Part.from_bytes(
                                data=image_bytes,
                                mime_type=mime_type
                            ))
                    else:
                        print(f"[Image Tools] Warning: Skipping invalid reference: {ref_image}")
                elif isinstance(ref_image, dict):
                    # Dict format with base64 data
                    if 'data' in ref_image:
                        # Decode base64 if it's a string
                        image_data = ref_image['data']
                        if isinstance(image_data, str):
                            image_data = base64.b64decode(image_data)
                        contents.append(types.Part.from_bytes(
                            data=image_data,
                            mime_type=ref_image.get('media_type', 'image/png')
                        ))

        # Add text prompt
        contents.append(prompt)

        # Get Gemini aspect ratio string
        gemini_aspect_ratio = GEMINI_ASPECT_RATIOS.get(aspect_ratio, "9:16")
        print(f"[Image Tools] Using aspect ratio: {aspect_ratio} ({gemini_aspect_ratio})")
        
        # Generate content with image modality and native aspect ratio support
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=gemini_aspect_ratio
                ),
                candidate_count=1,
                safety_settings=[
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
                ]
            )
        )
        
        # Process response
        result = {
            "success": False,
            "image_data": None,
            "text_response": None,
            "saved_path": None
        }
        
        # Extract response parts
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                # Skip reasoning/thought output from Gemini 3 Pro
                if hasattr(part, 'thought') and part.thought:
                    continue
                if hasattr(part, 'text') and part.text:
                    result["text_response"] = part.text
                elif hasattr(part, 'inline_data') and part.inline_data:
                    # Get image data
                    image_bytes = part.inline_data.data
                    result["image_data"] = base64.b64encode(image_bytes).decode('utf-8')
                    result["success"] = True
                    
                    # Optional aspect ratio validation (log only - native API should handle this)
                    if check_aspect_ratio:
                        actual_image = PILImage.open(BytesIO(image_bytes))
                        actual_width, actual_height = actual_image.size
                        actual_ratio = actual_width / actual_height
                        print(f"[Image Tools] Generated image: {actual_width}x{actual_height} (ratio {actual_ratio:.3f})")  
                    
                    # Always include image bytes in result for flexibility
                    result["image_bytes"] = image_bytes
                    
                    # Save image if path provided (backward compatibility)
                    if save_path:
                        try:
                            # Create directory if needed
                            save_dir = os.path.dirname(save_path)
                            if save_dir:
                                os.makedirs(save_dir, exist_ok=True)
                            
                            # Save the image
                            image = PILImage.open(BytesIO(image_bytes))
                            image.save(save_path)
                            result["saved_path"] = save_path
                                
                        except Exception as e:
                            result["save_error"] = f"Failed to save image: {str(e)}"
        
        if not result["success"]:
            result["error"] = "No image generated in response"
            
    except Exception as e:
        result = {
            "success": False,
            "error": f"Image generation error: {str(e)}",
            "image_data": None,
            "text_response": None,
            "saved_path": None
        }
    
    return result


def annotate_image(image_path: str) -> Dict[str, Any]:
    """
    Annotate a single image using Gemini multimodal capabilities
    
    Args:
        image_path: Path to the image file to annotate
        
    Returns:
        Dictionary containing:
        - description: Detailed visual description of the image
        - success: bool indicating if annotation succeeded
        - error: error message (if failed)
    """
    try:
        # Initialize Gemini client with global location for multimodal
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location="global",  # Use global location for multimodal
            credentials=CREDENTIALS
        )
        
        # Load image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Detect image type from extension
        ext = Path(image_path).suffix.lower()
        media_type = f"image/{ext[1:]}" if ext else "image/png"
        
        # Build content with image and analysis prompt
        contents = [
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=media_type
            ),
            """Analyze this image and provide a comprehensive description including:
            - Main subjects (people, characters, objects) - what is it?
            - Visual style and artistic technique
            - Colors, lighting, and mood
            - Clothing and appearance details
            - Setting and environment
            - Composition and framing
            - Any text or notable elements
            - Overall impression and potential use cases
            
            Focus on relevant fields from above to address and skip the fields that does not apply to this picture. 
            Be accurate and keep under 3 sentences. Be specific and detailed in your description."""
        ]
        
        # Generate analysis with Gemini 3 Pro (higher quality for detailed annotation)
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=2000,
                safety_settings=[
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
                ]
            )
        )
        
        # Extract description from response
        if response.candidates and response.candidates[0].content.parts:
            description = response.candidates[0].content.parts[0].text
            return {
                "success": True,
                "description": description
            }
        else:
            # Debug: Print raw response
            print(f"[Image Tools DEBUG] Raw response: {response}")
            
            return {
                "success": False,
                "error": "No response generated"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Annotation error: {str(e)}"
        }

def _annotate_single_image(image_url: str, has_gcs: bool, download_to_temp) -> Optional[Dict[str, Any]]:
    """
    Annotate a single image (helper for parallel processing)

    Args:
        image_url: GCS URL or local path to annotate
        has_gcs: Whether GCS utilities are available
        download_to_temp: GCS download function (if available)

    Returns:
        Annotation data dict or None if failed
    """
    from datetime import datetime

    try:
        # Handle GCS URLs vs local paths
        if image_url.startswith("gs://") or image_url.startswith("https://storage.googleapis.com"):
            # GCS URL - download temporarily
            if has_gcs and download_to_temp:
                with download_to_temp(image_url, suffix=".png") as temp_path:
                    result = annotate_image(temp_path)
            else:
                print(f"[Image Tools] Skipping GCS URL (no gcs_utils): {image_url}")
                return None
        else:
            # Local path - annotate directly
            if os.path.exists(image_url):
                result = annotate_image(image_url)
            else:
                print(f"[Image Tools] File not found: {image_url}")
                return None

        if result["success"]:
            return {
                "description": result["description"],
                "url": image_url,
                "timestamp": datetime.now().isoformat()
            }
        else:
            print(f"[Image Tools] Error annotating {image_url}: {result.get('error', 'Unknown error')}")
            return None

    except Exception as e:
        print(f"[Image Tools] Error annotating {image_url}: {str(e)}")
        return None


def get_or_create_annotations(reference_images: List[str], existing_annotations: Dict[str, Any] = None, state: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create annotations for reference images using parallel processing
    Downloads temporarily from GCS to annotate, then cleans up

    Args:
        reference_images: List of gs:// paths or local paths to annotate
        existing_annotations: Existing annotations to avoid re-processing
        state: RAGState for emitting streaming events (optional)

    Returns:
        Dictionary with filenames as keys and annotation data as values
    """
    # Start with existing annotations or empty dict
    annotations = existing_annotations or {}

    # Check if we have gcs_utils available
    try:
        from ...storage.gcs_utils import download_to_temp
        has_gcs = True
    except ImportError:
        has_gcs = False
        download_to_temp = None

    # Helper to extract filename from path
    def get_filename_key(path: str) -> str:
        """Extract filename from any path format"""
        # Remove query params if present
        clean_path = path.split('?')[0]
        # Get filename from path
        return Path(clean_path).name

    # Find images that need annotation
    images_to_annotate = []
    for image_url in reference_images:
        filename_key = get_filename_key(image_url)
        # Check if we already have annotation for this filename
        if filename_key not in annotations:
            images_to_annotate.append(image_url)

    # Annotate new images in parallel
    if images_to_annotate:
        print(f"[Image Tools] Found {len(images_to_annotate)} new images to annotate")
        print(f"[Image Tools] Using parallel processing with up to 15 concurrent workers")

        # Use ContextThreadPoolExecutor for context propagation (enables streaming from threads)
        from langchain_core.runnables.config import ContextThreadPoolExecutor
        # from concurrent.futures import as_completed  # Already imported at line 23
        completed_count = 0
        with ContextThreadPoolExecutor(max_workers=15) as executor:
            # Submit all annotation tasks
            future_to_url = {
                executor.submit(_annotate_single_image, url, has_gcs, download_to_temp): url
                for url in images_to_annotate
            }

            # Process results as they complete
            for future in as_completed(future_to_url):
                image_url = future_to_url[future]
                completed_count += 1

                try:
                    annotation_data = future.result()
                    if annotation_data:
                        # Use filename as key, store path inside annotation
                        filename_key = get_filename_key(image_url)
                        # Add gs:// path to annotation data
                        annotation_data["path"] = image_url
                        annotations[filename_key] = annotation_data
                        desc_len = len(annotation_data["description"])
                        print(f"[Image Tools] [{completed_count}/{len(images_to_annotate)}] Annotated: {filename_key} ({desc_len} chars)")

                        # Emit streaming event for real-time UI updates (matches character image pattern)
                        if state:
                            from ..base import emit_event
                            emit_event(state, "reference_image_annotated", {
                                "filename": filename_key,
                                "path": image_url,
                                "description": annotation_data["description"],
                                "timestamp": annotation_data.get("timestamp"),
                                "completed": completed_count,
                                "total": len(images_to_annotate)
                            }, agent_name="memory_manager")
                    else:
                        print(f"[Image Tools] [{completed_count}/{len(images_to_annotate)}] Error: {image_url}")
                except Exception as e:
                    print(f"[Image Tools] [{completed_count}/{len(images_to_annotate)}] Error processing {image_url}: {str(e)}")

        print(f"[Image Tools] Parallel annotation complete. Total annotations: {len(annotations)}")

    return annotations


def _generate_single_image(
    tool_name: str,
    prompt: str,
    reference_images: Optional[List[str]] = None,
    aspect_ratio: str = "vertical",
    session_id: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate a single image (helper for parallel processing)

    Args:
        tool_name: Image generation tool to use
        prompt: Text prompt for generation
        reference_images: Optional reference images
        aspect_ratio: Aspect ratio preset
        session_id: Session ID for tracking
        **kwargs: Additional tool-specific parameters

    Returns:
        Generation result dict with success, image_bytes, error
    """
    try:
        result = call_image_tool(
            tool_name=tool_name,
            prompt=prompt,
            reference_images=reference_images,
            aspect_ratio=aspect_ratio,
            session_id=session_id,
            **kwargs
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Image generation error: {str(e)}",
            "image_bytes": None,
            "image_data": None
        }


def batch_generate_images(
    prompts: List[Dict[str, Any]],
    max_workers: int = 12,
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Generate multiple images in parallel using ThreadPoolExecutor

    IMPORTANT: Only use for INDEPENDENT images with no dependencies.
    For dependent images (variations, iterations), use sequential generation.

    Args:
        prompts: List of generation configs, each containing:
            - tool_name: str (default: "nano_banana")
            - prompt: str (required)
            - reference_images: List[str] (optional)
            - aspect_ratio: str (default: "vertical")
            - session_id: str (optional)
            - **kwargs: Additional tool-specific parameters
        max_workers: Maximum concurrent workers (default: 12)
        progress_callback: Optional callback(completed_count, total_count, result)

    Returns:
        List of results in same order as input prompts

    Example Use Cases:
        - Independent: Multiple props, different concept arts, separate mood boards
        - Dependent (DON'T use batch): Variations of same item, iterative refinements
    """
    if not prompts:
        return []

    total_count = len(prompts)
    print(f"[Image Tools - Batch] Starting batch generation of {total_count} images")
    print(f"[Image Tools - Batch] Using up to {max_workers} concurrent workers")

    results = [None] * total_count
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all generation tasks with their index
        future_to_index = {}
        for idx, config in enumerate(prompts):
            # Extract config with defaults
            tool_name = config.get("tool_name", "nano_banana")
            prompt = config.get("prompt", "")
            reference_images = config.get("reference_images")
            aspect_ratio = config.get("aspect_ratio", "vertical")
            session_id = config.get("session_id")

            # Extract any additional kwargs
            extra_kwargs = {k: v for k, v in config.items()
                          if k not in ["tool_name", "prompt", "reference_images",
                                      "aspect_ratio", "session_id"]}

            future = executor.submit(
                _generate_single_image,
                tool_name=tool_name,
                prompt=prompt,
                reference_images=reference_images,
                aspect_ratio=aspect_ratio,
                session_id=session_id,
                **extra_kwargs
            )
            future_to_index[future] = idx

        # Process results as they complete
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            completed_count += 1

            try:
                result = future.result()
                results[idx] = result

                status = "SUCCESS" if result.get("success") else "FAILED"
                error_msg = f" - {result.get('error', '')}" if not result.get("success") else ""
                print(f"[Image Tools - Batch] [{completed_count}/{total_count}] Image {idx+1}: {status}{error_msg}")

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(completed_count, total_count, result)

            except Exception as e:
                results[idx] = {
                    "success": False,
                    "error": f"Execution error: {str(e)}",
                    "image_bytes": None,
                    "image_data": None
                }
                print(f"[Image Tools - Batch] [{completed_count}/{total_count}] Image {idx+1}: ERROR - {str(e)}")

                if progress_callback:
                    progress_callback(completed_count, total_count, results[idx])

    success_count = sum(1 for r in results if r and r.get("success"))
    print(f"[Image Tools - Batch] Batch generation complete: {success_count}/{total_count} succeeded")

    return results


# ==============================================================================
# TOOL-BASED IMAGE GENERATION FUNCTIONS
# ==============================================================================


def flux(
    prompt: str,
    reference_images: Optional[List[Union[str, Dict[str, Any]]]] = None,
    aspect_ratio: str = "vertical",
    session_id: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Flux: Alternative image generation tool (placeholder).
    Best for: Specific use cases, alternative style generation.

    Args:
        prompt: Text description for image generation
        reference_images: Optional list of reference images
        aspect_ratio: Aspect ratio preset
        session_id: Session ID for tracking
        **kwargs: Additional tool-specific parameters

    Returns:
        Dictionary with success, image_bytes, and optional error
    """
    # Placeholder for future implementation
    return {
        "success": False,
        "error": "flux tool not yet implemented - coming soon",
        "image_bytes": None,
        "image_data": None,
        "gcs_url": None
    }


# ==============================================================================
# SYNCHRONOUS UNIFIED IMAGE GENERATION
# ==============================================================================

def generate_or_edit_frame_sync(
    scene: int,
    shot: int,
    frame: int,
    prompt: str,
    reference_images: Optional[List[str]] = None,
    tool: str = "nano_banana",
    intermediate_name: Optional[str] = None,
    aspect_ratio: str = "horizontal",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Unified synchronous image generation function for storyboard sub-agents.

    For editing: include the image to edit in reference_images and describe
    the edit operation in the prompt.

    Args:
        scene: Scene number from script
        shot: Shot number within scene
        frame: Frame number (1, 2, etc.)
        prompt: Generation or editing prompt
        reference_images: Optional reference images (gs:// paths, max 14)
        tool: Image generation tool ("nano_banana")
        intermediate_name: Optional name suffix for intermediate operations
        aspect_ratio: Aspect ratio preset
        session_id: Session ID for GCS upload

    Returns:
        Dict with success, gcs_url, filename, operation, and reference metadata
    """
    from datetime import datetime

    # Build filename
    if intermediate_name:
        filename = f"sc{scene:02d}_sh{shot:02d}_fr{frame}_{intermediate_name}.png"
    else:
        filename = f"sc{scene:02d}_sh{shot:02d}_fr{frame}.png"

    print(f"\n[Sync Image Tools] ========== FRAME GENERATION START ==========")
    print(f"[Sync Image Tools] Scene: {scene}, Shot: {shot}, Frame: {frame}")
    print(f"[Sync Image Tools] Filename: {filename}")
    print(f"[Sync Image Tools] Tool: {tool}")
    print(f"[Sync Image Tools] Intermediate: {intermediate_name if intermediate_name else 'None (final)'}")
    if intermediate_name:
        print(f"[Sync Image Tools] [CHAINING] Step 1 detected - intermediate name: {intermediate_name}")
    print(f"[Sync Image Tools] References: {len(reference_images) if reference_images else 0}")
    if reference_images:
        for idx, ref in enumerate(reference_images, 1):
            ref_name = ref.split('/')[-1] if '/' in ref else ref
            print(f"[Sync Image Tools]   Ref {idx}: {ref_name}")
    print(f"[Sync Image Tools] Prompt preview: {prompt[:100]}...")

    # Validate reference count (upgraded limit: 14 total)
    if reference_images and len(reference_images) > 14:
        return {
            "success": False,
            "error": f"Too many references: {len(reference_images)} provided, maximum is 14",
            "filename": filename
        }

    # NOTE: Aspect ratio validation removed - image APIs support multiple ratios (16:9, 9:16, 1:1, 4:3, 3:4, 21:9, etc.)
    # Video generation validation happens at video submission layer, not image generation
    # VALID_VIDEO_ASPECT_RATIOS = ["horizontal", "vertical"]
    # if aspect_ratio not in VALID_VIDEO_ASPECT_RATIOS:
    #     return {
    #         "success": False,
    #         "error": f"Invalid aspect_ratio '{aspect_ratio}' for video generation. "
    #                f"Must be 'horizontal' (16:9) or 'vertical' (9:16). "
    #                f"Video providers require these specific ratios.",
    #         "filename": filename
    #     }

    try:
        if tool == "nano_banana":
            # Nano Banana is already synchronous
            print(f"[Sync Image Tools] Using Nano Banana (synchronous)")

            result = nano_banana(
                prompt=prompt,
                reference_images=reference_images,
                aspect_ratio=aspect_ratio,
                session_id=session_id
            )

            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error", "Nano Banana generation failed"),
                    "filename": filename
                }

            # Upload to GCS (returns gs:// path by default)
            gcs_url = None
            if session_id and result.get("image_bytes"):
                try:
                    from ...storage.gcs_utils import upload_bytes_to_gcs
                    from ...core.config import ENABLE_ASPECT_RATIO_CORRECTION
                    from .util_image import correct_aspect_ratio

                    # Apply aspect ratio correction if enabled (storyboard-only, final frames only)
                    final_image_bytes = result["image_bytes"]
                    if ENABLE_ASPECT_RATIO_CORRECTION and intermediate_name is None:
                        final_image_bytes = correct_aspect_ratio(final_image_bytes, aspect_ratio)

                    print(f"[Sync Image Tools] Uploading to GCS ({len(final_image_bytes)} bytes)...")
                    gcs_url = upload_bytes_to_gcs(
                        data=final_image_bytes,
                        session_id=session_id,
                        asset_type="storyboards",
                        filename=filename,
                        content_type="image/png",
                        return_format="gs"
                    )
                    print(f"[Sync Image Tools] SUCCESS - Uploaded: {gcs_url}")
                except Exception as e:
                    print(f"[Sync Image Tools] ERROR - GCS upload error: {str(e)}")
                    return {
                        "success": False,
                        "error": f"GCS upload error: {str(e)}",
                        "filename": filename
                    }

            print(f"[Sync Image Tools] ========== FRAME GENERATION COMPLETE ==========\n")
            return {
                "success": True,
                "gcs_url": gcs_url or "",
                "filename": filename,
                "tool": tool,
                "timestamp": datetime.now().isoformat(),
                "reference_images": reference_images,  # For UI display
                "file_uri": gcs_url or "",  # For multimodal function responses
                "mime_type": "image/png"  # For multimodal function responses
            }

        else:
            return {
                "success": False,
                "error": f"Unknown tool: {tool}",
                "filename": filename
            }

    except Exception as e:
        print(f"[Sync Image Tools] Exception: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "filename": filename
        }


# Tool registry for dispatch
IMAGE_TOOLS = {
    "nano_banana": nano_banana,
    "flux": flux
}


def call_image_tool(tool_name: str = "nano_banana", **kwargs) -> Dict[str, Any]:
    """
    Simple dispatcher for image generation tools.

    Args:
        tool_name: Name of the image generation tool (default: nano_banana)
        **kwargs: Tool-specific parameters

    Returns:
        Tool execution result
    """
    if tool_name not in IMAGE_TOOLS:
        return {
            "success": False,
            "error": f"Unknown image tool: {tool_name}",
            "image_bytes": None,
            "image_data": None,
            "gcs_url": None
        }

    return IMAGE_TOOLS[tool_name](**kwargs)


# ==============================================================================
# LEGACY WRAPPER (for backward compatibility)
# ==============================================================================

def generate_image(
    prompt: str,
    reference_images: Optional[List[Union[str, Dict[str, Any]]]] = None,
    save_path: Optional[str] = None,
    reference_metadata: Optional[Dict[str, Any]] = None,
    aspect_ratio: str = "vertical",
    check_aspect_ratio: bool = False,
    tool_name: str = "nano_banana"
) -> Dict[str, Any]:
    """
    Legacy wrapper function for backward compatibility.
    New code should use call_image_tool() or tool functions directly.

    Args:
        Same as nano_banana, plus:
        tool_name: Which tool to use (default: nano_banana)

    Returns:
        Tool execution result
    """
    return call_image_tool(
        tool_name=tool_name,
        prompt=prompt,
        reference_images=reference_images,
        aspect_ratio=aspect_ratio,
        save_path=save_path,
        reference_metadata=reference_metadata,
        check_aspect_ratio=check_aspect_ratio
    )