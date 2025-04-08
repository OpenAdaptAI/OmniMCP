# omnimcp/utils.py

"""Minimal utilities needed for OmniMCP."""

from functools import wraps
from io import BytesIO
from typing import Any, Callable, List, Tuple, Union, Optional
import base64
import os
import sys
import threading
import time
import textwrap

from jinja2 import Environment, Template
from loguru import logger
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import mss

if sys.platform == "darwin":
    try:
        from AppKit import NSScreen
    except ImportError:
        logger.error(
            "AppKit not found. Install it with 'pip install pyobjc-framework-Cocoa' for proper scaling on macOS."
        )
        NSScreen = None
else:
    NSScreen = None  # Define as None on other platforms

from .types import UIElement, LLMActionPlan

# Process-local storage for MSS instances
_process_local = threading.local()


def get_process_local_sct() -> mss.mss:
    """Retrieve or create the `mss` instance for the current process."""
    if not hasattr(_process_local, "sct"):
        _process_local.sct = mss.mss()
    return _process_local.sct


def take_screenshot() -> Image.Image:
    """Take a screenshot of the entire screen.

    Returns:
        PIL.Image.Image: The screenshot image
    """
    sct = get_process_local_sct()
    monitor = sct.monitors[0]
    sct_img = sct.grab(monitor)
    image = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    return image


def get_monitor_dims() -> Tuple[int, int]:
    """Get the dimensions reported by mss for the primary monitor."""
    # This might return logical points or physical pixels depending on backend/OS.
    # The scaling factor helps bridge the gap regardless.
    sct = get_process_local_sct()
    monitor_index = (
        1 if len(sct.monitors) > 1 else 0
    )  # Use primary monitor (often index 1)
    monitor = sct.monitors[monitor_index]
    dims = (monitor["width"], monitor["height"])
    logger.debug(f"mss reported monitor dims: {dims}")
    return dims


def image_to_base64(image: Union[str, Image.Image]) -> str:
    """Convert image to base64 string.

    Args:
        image: PIL Image or path to image file

    Returns:
        str: Base64 encoded image string
    """
    if isinstance(image, str):
        image = Image.open(image)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def log_action(func: Callable) -> Callable:
    """Decorator to log function calls with timing."""

    @wraps(func)
    def wrapper(*args: tuple, **kwargs: dict) -> Any:
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = (time.time() - start) * 1000
            logger.debug(f"{func.__name__} completed in {duration:.2f}ms")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {str(e)}")
            raise

    return wrapper


def denormalize_coordinates(
    norm_x: float,
    norm_y: float,
    screen_w: int,
    screen_h: int,  # These are PHYSICAL PIXEL dimensions of the screenshot
    norm_w: Optional[float] = None,
    norm_h: Optional[float] = None,
) -> Tuple[int, int]:
    """
    Convert normalized coordinates (relative to screenshot) to
    ABSOLUTE PHYSICAL PIXEL coordinates.
    """
    if screen_w <= 0 or screen_h <= 0:
        return 0, 0
    if norm_w is not None and norm_h is not None:
        center_x_norm = norm_x + norm_w / 2
        center_y_norm = norm_y + norm_h / 2
        abs_x = int(center_x_norm * screen_w)
        abs_y = int(center_y_norm * screen_h)
    else:
        abs_x = int(norm_x * screen_w)
        abs_y = int(norm_y * screen_h)
    abs_x = max(0, min(screen_w - 1, abs_x))
    abs_y = max(0, min(screen_h - 1, abs_y))
    return abs_x, abs_y


def normalize_coordinates(
    x: int, y: int, screen_w: int, screen_h: int
) -> Tuple[float, float]:
    if screen_w <= 0 or screen_h <= 0:
        logger.warning(
            f"Invalid screen dimensions ({screen_w}x{screen_h}), cannot normalize."
        )
        return 0.0, 0.0
    norm_x = max(0.0, min(1.0, x / screen_w))
    norm_y = max(0.0, min(1.0, y / screen_h))
    return norm_x, norm_y


def get_scale_ratios() -> Tuple[float, float]:
    """Get the scale ratios between actual screen dimensions and image dimensions.

    Handles high DPI/Retina displays where screenshot dimensions may differ from
    logical screen dimensions.

    Returns:
        tuple[float, float]: The (width_ratio, height_ratio) scaling factors.

    Example:
        # Convert screen coordinates to image coordinates
        width_ratio, height_ratio = get_scale_ratios()
        image_x = screen_x * width_ratio
        image_y = screen_y * height_ratio
    """
    # Get actual screen dimensions
    monitor_width, monitor_height = get_monitor_dims()

    # Get dimensions of screenshot image
    image = take_screenshot()

    # Calculate scaling ratios
    width_ratio = image.width / monitor_width
    height_ratio = image.height / monitor_height

    logger.debug(f"Scale ratios - width: {width_ratio:.2f}, height: {height_ratio:.2f}")
    return width_ratio, height_ratio


def screen_to_image_coords(x: int, y: int) -> Tuple[int, int]:
    """Convert screen coordinates to image coordinates.

    Args:
        x: Screen x coordinate
        y: Screen y coordinate

    Returns:
        tuple[int, int]: The (x, y) coordinates in image space
    """
    width_ratio, height_ratio = get_scale_ratios()
    image_x = int(x * width_ratio)
    image_y = int(y * height_ratio)
    return image_x, image_y


def image_to_screen_coords(x: int, y: int) -> Tuple[int, int]:
    """Convert image coordinates to screen coordinates.

    Args:
        x: Image x coordinate
        y: Image y coordinate

    Returns:
        tuple[int, int]: The (x, y) coordinates in screen space
    """
    width_ratio, height_ratio = get_scale_ratios()
    screen_x = int(x / width_ratio)
    screen_y = int(y / height_ratio)
    return screen_x, screen_y


def retry_with_exceptions(max_retries: int = 5) -> Callable:
    """Decorator to retry a function while keeping track of exceptions."""

    def decorator_retry(func: Callable) -> Callable:
        @wraps(func)
        def wrapper_retry(*args: tuple, **kwargs: dict[str, Any]) -> Any:
            exceptions = []
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    logger.warning(exc)
                    exceptions.append(str(exc))
                    retries += 1
                    last_exception = exc
            raise RuntimeError(
                f"Failed after {max_retries} retries with exceptions: {exceptions}"
            ) from last_exception

        return wrapper_retry

    return decorator_retry


def compute_diff(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Computes the difference between two PIL Images.

    Useful for verifying UI changes after actions.

    Returns:
        PIL.Image.Image: Difference image showing changed pixels
    """
    import numpy as np

    arr1 = np.array(image1)
    arr2 = np.array(image2)
    diff = np.abs(arr1 - arr2)
    return Image.fromarray(diff.astype("uint8"))


def increase_contrast(image: Image.Image, contrast_factor: float = 1.5) -> Image.Image:
    """Increase the contrast of an image to help with UI element detection.

    Args:
        image: The image to enhance
        contrast_factor: Values > 1 increase contrast, < 1 decrease it

    Returns:
        Enhanced image
    """
    from PIL import ImageEnhance

    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(contrast_factor)


def create_prompt_template(template_str: str) -> Template:
    """Create a Jinja2 template from a multiline string.

    Handles proper dedenting and whitespace cleanup for clear template definitions.

    Args:
        template_str: Raw template string, can be triple-quoted multiline

    Returns:
        Jinja2 Template object ready for rendering

    Example:
        template = create_prompt_template('''
            Analyze this UI element and describe its properties:

            Screenshot context:
            {{ screenshot_desc }}

            Element bounds:
            {% for coord in coordinates %}
            - {{ coord }}
            {% endfor %}

            Previous actions:
            {% for action in action_history %}
            {{ loop.index }}. {{ action }}
            {% endfor %}
        ''')

        prompt = template.render(
            screenshot_desc=desc,
            coordinates=coords,
            action_history=history
        ).strip()
    """
    # Create Jinja environment with proper whitespace handling
    env = Environment(
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Dedent the template string to remove common leading whitespace
    template_str = textwrap.dedent(template_str)

    # Create and return the template
    return env.from_string(template_str)


def render_prompt(template: Union[Template, str], **kwargs: Any) -> str:
    """Create and render a prompt template in one step.

    Args:
        template_str: Raw template string
        **kwargs: Variables to render into template

    Returns:
        Rendered prompt string

    Example:
        prompt = render_prompt('''
            Analyze this UI element:
            {{ element_desc }}

            Coordinates: {% for c in coords %}{{ c }} {% endfor %}
        ''',
        element_desc="Blue button with text 'Submit'",
        coords=[10, 20, 30, 40]
        )
    """
    if isinstance(template, str):
        template = create_prompt_template(template)
    try:
        return template.render(**kwargs).strip()
    except Exception as e:
        logger.error(f"Error rendering prompt template: {e}")
        logger.debug(f"Template variables: {kwargs}")
        raise


def draw_bounding_boxes(
    image: Image.Image,
    elements: List["UIElement"],
    color: str = "red",
    width: int = 1,
    show_ids: bool = True,
) -> Image.Image:
    """
    Draws bounding boxes and optionally IDs for a list of UIElements onto an image.

    Args:
        image: The PIL Image to draw on.
        elements: A list of UIElement objects.
        color: Color of the bounding boxes and text.
        width: Width of the bounding box lines.
        show_ids: Whether to draw the element ID text.

    Returns:
        A new PIL Image with the drawings. Returns original if errors occur.
    """
    if not elements:
        return image.copy()  # Return a copy if no elements

    try:
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        # Try to load a basic font, fallback to default
        try:
            # Adjust font path/size as needed, or use a default PIL font
            # font = ImageFont.truetype("arial.ttf", 12) # Might fail if not installed
            font_size = 12
            font = ImageFont.load_default(size=font_size)
        except IOError:
            logger.warning(
                "Default font not found for drawing IDs. Using basic PIL font."
            )
            font = ImageFont.load_default()
            font_size = 10  # Default font might be larger

        img_width, img_height = image.size

        for element in elements:
            try:
                # Denormalize bounds (x, y, w, h) -> (x1, y1, x2, y2)
                x1 = int(element.bounds[0] * img_width)
                y1 = int(element.bounds[1] * img_height)
                x2 = int((element.bounds[0] + element.bounds[2]) * img_width)
                y2 = int((element.bounds[1] + element.bounds[3]) * img_height)

                # Clamp coordinates to image boundaries
                x1 = max(0, min(img_width - 1, x1))
                y1 = max(0, min(img_height - 1, y1))
                x2 = max(0, min(img_width, x2))  # Allow x2/y2 to be == width/height
                y2 = max(0, min(img_height, y2))

                # Ensure coordinates are valid (x1 < x2, y1 < y2)
                if x1 >= x2 or y1 >= y2:
                    logger.warning(
                        f"Skipping drawing element ID {element.id} due to invalid coords after denormalization: ({x1},{y1})-({x2},{y2})"
                    )
                    continue

                # Draw rectangle
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=width)

                # Draw ID text
                if show_ids:
                    text = str(element.id)
                    # Simple positioning near top-left corner
                    text_x = x1 + width + 1
                    text_y = y1 + width + 1

                    # Basic check to keep text within bounds (doesn't handle long text well)
                    if text_x < img_width - 10 and text_y < img_height - font_size - 1:
                        # Simple background rectangle for visibility
                        # text_bbox = draw.textbbox((text_x, text_y), text, font=font)
                        # draw.rectangle(text_bbox, fill=(255,255,255,180)) # Semi-transparent white bg
                        draw.text((text_x, text_y), text, fill=color, font=font)

            except Exception as el_draw_e:
                logger.warning(f"Error drawing element ID {element.id}: {el_draw_e}")
                continue  # Skip this element

        return draw_image

    except Exception as e:
        logger.error(f"Failed to draw bounding boxes: {e}", exc_info=True)
        return image.copy()  # Return a copy of original on major error


def get_scaling_factor() -> int:
    """
    Determine the display scaling factor (e.g., 2 for Retina).
    Uses AppKit on macOS, defaults to 1 otherwise.
    """
    if sys.platform == "darwin" and NSScreen:
        try:
            # Get the scale factor from the main screen
            backing_scale = NSScreen.mainScreen().backingScaleFactor()
            logger.debug(f"Detected macOS backingScaleFactor: {backing_scale}")
            return int(backing_scale)
        except Exception as e:
            logger.error(
                f"Error getting macOS backingScaleFactor: {e}. Defaulting to 1."
            )
            return 1
    else:
        # Default for non-macOS platforms or if AppKit failed
        logger.debug("Not on macOS or AppKit unavailable, using scaling factor 1.")
        return 1


# Attempt to load a common font, with fallback
try:
    # Adjust size as needed
    ACTION_FONT = ImageFont.truetype("arial.ttf", 14)
except IOError:
    logger.warning("Arial font not found for highlighting. Using default PIL font.")
    ACTION_FONT = ImageFont.load_default()


def draw_action_highlight(
    image: Image.Image,
    element: UIElement,
    plan: LLMActionPlan,
    color: str = "red",
    width: int = 3,
    dim_factor: float = 0.5,
    text_color: str = "black",
    text_bg_color: Tuple[int, int, int, int] = (255, 255, 255, 200),  # White with alpha
) -> Image.Image:
    """
    Draws highlight box, dims background, and adds text annotation for the planned action,
    using the actual image dimensions for coordinate calculations.

    Args:
        image: The source PIL Image (e.g., the screenshot).
        element: The UIElement targeted by the action.
        plan: The LLMActionPlan object for the action.
        color: Color of the highlight box.
        width: Line width of the highlight box.
        dim_factor: Background dimming factor (0.0 to 1.0).
        text_color: Annotation text color.
        text_bg_color: Annotation text background color (RGBA tuple).

    Returns:
        A new PIL Image with the highlight and annotation.
    """
    if not image or not plan:
        logger.warning("draw_action_highlight: Missing image or plan.")
        # Return a copy to avoid modifying original if subsequent steps fail
        return (
            image.copy() if image else Image.new("RGB", (100, 50))
        )  # Placeholder image

    final_image = image.copy()
    img_width, img_height = image.size
    draw = ImageDraw.Draw(final_image)
    margin = 5

    try:
        # --- Draw Box and Dim Background ONLY if element is present ---
        if element and hasattr(element, "bounds"):
            # Denormalize using actual image dimensions
            abs_x, abs_y = denormalize_coordinates(
                element.bounds[0], element.bounds[1], img_width, img_height
            )
            abs_w = int(element.bounds[2] * img_width)
            abs_h = int(element.bounds[3] * img_height)
            x0, y0 = max(0, abs_x), max(0, abs_y)
            x1, y1 = min(img_width, abs_x + abs_w), min(img_height, abs_y + abs_h)
            element_box = (x0, y0, x1, y1)

            # Apply Dimming
            if 0.0 <= dim_factor < 1.0:
                try:
                    enhancer = ImageEnhance.Brightness(final_image)
                    dimmed_image = enhancer.enhance(dim_factor)
                    if x0 < x1 and y0 < y1:  # Ensure valid crop box
                        original_element_area = image.crop(element_box)
                        dimmed_image.paste(original_element_area, (x0, y0))
                    final_image = dimmed_image
                except Exception as dim_e:
                    logger.warning(f"Could not apply dimming effect: {dim_e}")

            # Draw Highlight Box
            if x0 < x1 and y0 < y1:  # Ensure valid box
                draw.rectangle(element_box, outline=color, width=width)
        # --- End Element-Specific Drawing ---

        # --- Always Draw Text Annotation ---
        try:
            action_text = str(plan.action).capitalize()
            details = ""
            if plan.action == "type" and plan.text_to_type is not None:
                text_preview = (
                    (plan.text_to_type[:20] + "...")
                    if len(plan.text_to_type) > 23
                    else plan.text_to_type
                )
                details = f"'{text_preview}'"
            elif plan.action == "press_key" and plan.key_info:
                details = f"'{plan.key_info}'"
            elif plan.action == "click" and element:
                details = f"on ID {element.id}"  # Add element ID for click clarity

            annotation_text = f"Next: {action_text} {details}".strip()

            # Calculate text size
            try:
                text_bbox = draw.textbbox((0, 0), annotation_text, font=ACTION_FONT)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_width, text_height = (
                    draw.textlength(annotation_text, font=ACTION_FONT),
                    ACTION_FONT.getbbox("A")[3] + 2,
                )

            # Position: Top-left if no element, otherwise above element box
            if element and hasattr(element, "bounds"):
                text_x = max(
                    margin,
                    min(
                        x0 + (abs_w - text_width) // 2, img_width - text_width - margin
                    ),
                )
                text_y = max(margin, y0 - text_height - margin)
            else:  # No target element, put text at top-left
                text_x = margin
                text_y = margin

            # Draw background rectangle
            bg_x0, bg_y0 = max(0, text_x - margin // 2), max(0, text_y - margin // 2)
            bg_x1, bg_y1 = (
                min(img_width, text_x + text_width + margin // 2),
                min(img_height, text_y + text_height + margin // 2),
            )
            if bg_x0 < bg_x1 and bg_y0 < bg_y1:
                draw.rectangle([(bg_x0, bg_y0), (bg_x1, bg_y1)], fill=text_bg_color)

            # Draw text
            draw.text(
                (text_x, text_y), annotation_text, fill=text_color, font=ACTION_FONT
            )

        except Exception as text_e:
            logger.warning(f"Failed to draw text annotation: {text_e}")
        # --- End Text Annotation ---

    except Exception as e:
        logger.error(f"Failed during drawing highlight: {e}", exc_info=True)
        return image.copy()  # Return copy of original on error

    return final_image


def downsample_image(image: Image.Image, factor: float) -> Image.Image:
    """
    Downsamples a PIL Image by a given factor using LANCZOS resampling.

    Args:
        image: The original PIL Image.
        factor: The scaling factor (e.g., 0.5 for 50%). Must be > 0 and <= 1.

    Returns:
        The downsampled PIL Image. Returns original if factor is invalid or error occurs.
    """
    if not (0.0 < factor <= 1.0):
        logger.warning(
            f"Invalid downsample factor ({factor}). Returning original image."
        )
        return image
    if factor == 1.0:
        return image  # No scaling needed

    try:
        original_dimensions = image.size
        new_width = int(original_dimensions[0] * factor)
        new_height = int(original_dimensions[1] * factor)
        # Ensure dimensions are at least 1x1
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        start_time = time.time()
        # Use LANCZOS for potentially better quality downsampling
        scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        duration = (time.time() - start_time) * 1000
        logger.debug(
            f"Resized image from {original_dimensions} to {scaled_image.size} (factor {factor:.2f}) in {duration:.1f}ms"
        )
        return scaled_image
    except Exception as resize_err:
        logger.warning(f"Failed to downsample image, returning original: {resize_err}")
        return image  # Fallback to original on error


def setup_run_logging(run_dir=None):
    """
    Configure stderr and optional file logging for a specific run.

    Removes default handlers and sets up new ones based on config and run_dir.

    Args:
        run_dir: Directory to store run-specific logs. If None, only stderr logging
                 at the configured level is guaranteed (unless default file logging is enabled).

    Returns:
        The log file path if file logging was configured, otherwise None.
    """
    from omnimcp.config import config

    # Remove default loguru handler to avoid duplicate messages if added by default
    try:
        logger.remove(0)  # Attempt to remove the default handler ID 0
    except ValueError:
        logger.warning(
            "Could not remove default logger handler (ID 0). May already be removed."
        )
        pass  # Ignore if it was already removed or never added

    # Configure stderr logging based on config
    stderr_level = config.LOG_LEVEL.upper() if config.LOG_LEVEL else "INFO"
    logger.add(sys.stderr, level=stderr_level)
    logger.debug(f"Configured stderr logging level: {stderr_level}")

    # Configure file logging if run_dir is provided
    log_file_path = None
    if run_dir:
        try:
            os.makedirs(run_dir, exist_ok=True)
            log_file_path = os.path.join(run_dir, "run.log")
            # Add run-specific log handler (level DEBUG for file)
            logger.add(
                log_file_path,
                rotation="50 MB",
                level="DEBUG",
                encoding="utf8",
                enqueue=True,
            )
            logger.info(
                f"Configured run-specific file logging. Log path: {log_file_path}"
            )
        except Exception as e:
            logger.error(
                f"Failed to configure file logging for run_dir '{run_dir}': {e}"
            )
            log_file_path = None  # Ensure path is None on failure
    elif not config.DISABLE_DEFAULT_LOGGING and config.LOG_DIR:
        # Fallback to default log directory if run_dir is None and default logging not disabled
        try:
            default_log_dir = config.LOG_DIR
            os.makedirs(default_log_dir, exist_ok=True)
            log_file_path = os.path.join(
                default_log_dir, "omnimcp_{time:YYYY-MM-DD}.log"
            )  # Daily default log
            logger.add(
                log_file_path,
                rotation="1 day",
                level="DEBUG",
                encoding="utf8",
                enqueue=True,
            )
            logger.info(
                f"Configured default file logging. Log path pattern: {log_file_path}"
            )
        except Exception as e:
            logger.error(
                f"Failed to configure default file logging in '{config.LOG_DIR}': {e}"
            )
            log_file_path = None
    else:
        logger.info("File logging disabled or no directory specified.")

    return log_file_path  # Return path or None
