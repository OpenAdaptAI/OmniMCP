# omnimcp/synthetic_ui.py
import copy  # For deep copying element list
import os
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from .types import Bounds, UIElement
from .utils import logger

# --- Constants and Font ---
IMG_WIDTH, IMG_HEIGHT = 800, 600
try:
    FONT = ImageFont.truetype("arial.ttf", 15)
    FONT_BOLD = ImageFont.truetype("arialbd.ttf", 20)  # Added bold font
except OSError:
    logger.warning("Arial fonts not found. Using default PIL font.")
    FONT = ImageFont.load_default()
    FONT_BOLD = ImageFont.load_default()


# --- Coordinate Conversion ---
def _bounds_to_abs(bounds: Bounds) -> tuple[int, int, int, int]:
    """Convert normalized bounds to absolute pixel coordinates."""
    x, y, w, h = bounds
    abs_x = int(x * IMG_WIDTH)
    abs_y = int(y * IMG_HEIGHT)
    abs_w = int(w * IMG_WIDTH)
    abs_h = int(h * IMG_HEIGHT)
    return abs_x, abs_y, abs_w, abs_h


def _abs_to_bounds(abs_coords: tuple[int, int, int, int]) -> Bounds:
    """Convert absolute pixel coordinates to normalized bounds."""
    abs_x, abs_y, abs_w, abs_h = abs_coords
    x = abs_x / IMG_WIDTH
    y = abs_y / IMG_HEIGHT
    w = abs_w / IMG_WIDTH
    h = abs_h / IMG_HEIGHT
    return x, y, w, h


# --- UI Generation ---


def generate_login_screen(
    save_path: str | None = None,
) -> tuple[Image.Image, list[UIElement]]:
    """Generates the initial synthetic login screen image and element data."""
    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)
    elements: list[UIElement] = []
    element_id_counter = 0

    # Title
    title_text = "Welcome Back!"
    title_bbox = draw.textbbox((0, 0), title_text, font=FONT)
    title_w, _title_h = title_bbox[2] - title_bbox[0], title_bbox[3] - title_bbox[1]
    title_x, title_y = (IMG_WIDTH - title_w) / 2, 80
    draw.text((title_x, title_y), title_text, fill="black", font=FONT)

    # Username Field
    uname_label_y, uname_x, uname_w, uname_h = 150, 200, 400, 40
    uname_field_y = uname_label_y + 25
    draw.text((uname_x, uname_label_y), "Username:", fill="black", font=FONT)
    draw.rectangle(
        [(uname_x, uname_field_y), (uname_x + uname_w, uname_field_y + uname_h)],
        fill="white",
        outline="black",
    )
    elements.append(
        UIElement(
            id=element_id_counter,
            type="text_field",
            content="",
            bounds=_abs_to_bounds((uname_x, uname_field_y, uname_w, uname_h)),
            attributes={"label": "Username:"},  # Store label for potential use
        )
    )
    element_id_counter += 1

    # Password Field
    pw_label_y = uname_field_y + uname_h + 20
    pw_x, pw_w, pw_h = 200, 400, 40
    pw_field_y = pw_label_y + 25
    draw.text((pw_x, pw_label_y), "Password:", fill="black", font=FONT)
    draw.rectangle(
        [(pw_x, pw_field_y), (pw_x + pw_w, pw_field_y + pw_h)],
        fill="white",
        outline="black",
    )
    elements.append(
        UIElement(
            id=element_id_counter,
            type="text_field",
            content="",
            bounds=_abs_to_bounds((pw_x, pw_field_y, pw_w, pw_h)),
            attributes={"is_password": True, "label": "Password:"},
        )
    )
    element_id_counter += 1

    # Remember Me Checkbox
    cb_y = pw_field_y + pw_h + 30
    cb_x, cb_size = 200, 20
    cb_text_x = cb_x + cb_size + 10
    draw.rectangle(
        [(cb_x, cb_y), (cb_x + cb_size, cb_y + cb_size)], fill="white", outline="black"
    )
    draw.text((cb_text_x, cb_y + 2), "Remember Me", fill="black", font=FONT)
    elements.append(
        UIElement(
            id=element_id_counter,
            type="checkbox",
            content="Remember Me",
            bounds=_abs_to_bounds((cb_x, cb_y, cb_size, cb_size)),
            attributes={"checked": False},
        )
    )
    element_id_counter += 1

    # Forgot Password Link
    fp_text = "Forgot Password?"
    fp_bbox = draw.textbbox((0, 0), fp_text, font=FONT)
    fp_w, fp_h = fp_bbox[2] - fp_bbox[0], fp_bbox[3] - fp_bbox[1]
    fp_x, fp_y = pw_x + pw_w - fp_w, cb_y + 5
    draw.text((fp_x, fp_y), fp_text, fill="blue", font=FONT)
    elements.append(
        UIElement(
            id=element_id_counter,
            type="link",
            content="Forgot Password?",
            bounds=_abs_to_bounds((fp_x, fp_y, fp_w, fp_h)),
        )
    )
    element_id_counter += 1

    # Login Button
    btn_y = cb_y + cb_size + 40
    btn_w, btn_h = 120, 45
    btn_x = (IMG_WIDTH - btn_w) / 2
    draw.rectangle(
        [(btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h)], fill="green", outline="black"
    )
    btn_text = "Login"
    btn_bbox = draw.textbbox((0, 0), btn_text, font=FONT)
    btn_text_w, btn_text_h = btn_bbox[2] - btn_bbox[0], btn_bbox[3] - btn_bbox[1]
    draw.text(
        (btn_x + (btn_w - btn_text_w) / 2, btn_y + (btn_h - btn_text_h) / 2),
        btn_text,
        fill="white",
        font=FONT,
    )
    elements.append(
        UIElement(
            id=element_id_counter,
            type="button",
            content="Login",
            bounds=_abs_to_bounds((btn_x, btn_y, btn_w, btn_h)),
        )
    )
    element_id_counter += 1

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        img.save(save_path)
        logger.info(f"Saved synthetic UI to {save_path}")

    return img, elements


def generate_logged_in_screen(
    username: str, save_path: str | None = None
) -> tuple[Image.Image, list[UIElement]]:
    """Generates a simple 'logged in' screen."""
    img = Image.new(
        "RGB", (IMG_WIDTH, IMG_HEIGHT), color=(210, 230, 210)
    )  # Light green background
    draw = ImageDraw.Draw(img)
    elements: list[UIElement] = []
    element_id_counter = 0  # Start fresh IDs for new screen state

    # Welcome Message
    welcome_text = f"Welcome, {username}!"
    welcome_bbox = draw.textbbox((0, 0), welcome_text, font=FONT_BOLD)
    welcome_w, welcome_h = (
        welcome_bbox[2] - welcome_bbox[0],
        welcome_bbox[3] - welcome_bbox[1],
    )
    welcome_x, welcome_y = (IMG_WIDTH - welcome_w) / 2, 200
    draw.text((welcome_x, welcome_y), welcome_text, fill="darkgreen", font=FONT_BOLD)
    elements.append(
        UIElement(
            id=element_id_counter,
            type="text",
            content=welcome_text,
            bounds=_abs_to_bounds(
                (int(welcome_x), int(welcome_y), welcome_w, welcome_h)
            ),
            attributes={"is_heading": True},
        )
    )
    element_id_counter += 1

    # Logout Button
    btn_y = welcome_y + welcome_h + 50
    btn_w, btn_h = 120, 45
    btn_x = (IMG_WIDTH - btn_w) / 2
    draw.rectangle(
        [(btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h)], fill="orange", outline="black"
    )
    btn_text = "Logout"
    btn_bbox = draw.textbbox((0, 0), btn_text, font=FONT)
    btn_text_w, btn_text_h = btn_bbox[2] - btn_bbox[0], btn_bbox[3] - btn_bbox[1]
    draw.text(
        (btn_x + (btn_w - btn_text_w) / 2, btn_y + (btn_h - btn_text_h) / 2),
        btn_text,
        fill="black",
        font=FONT,
    )
    elements.append(
        UIElement(
            id=element_id_counter,
            type="button",
            content="Logout",
            bounds=_abs_to_bounds((int(btn_x), int(btn_y), btn_w, btn_h)),
        )
    )
    element_id_counter += 1

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        img.save(save_path)
        logger.info(f"Saved 'Logged In' screen to {save_path}")

    return img, elements


# --- Simulation Logic ---


def simulate_action(
    image: Image.Image,
    elements: list[UIElement],
    plan: Any,  # Using Any to avoid circular import with core.py/LLMActionPlan
    username_for_login: str = "User",  # Default username for welcome screen
) -> tuple[Image.Image, list[UIElement]]:
    """
    Simulates the effect of a planned action on the synthetic UI state.

    Args:
        image: The current PIL Image.
        elements: The current list of UIElements.
        plan: The LLMActionPlan object for the action to simulate.
        username_for_login: Username to display on successful login screen.

    Returns:
        A tuple containing the new (PIL Image, List[UIElement]) after simulation.
        Returns the original state if action cannot be simulated.
    """
    if plan.is_goal_complete:
        logger.info("Goal is marked complete, no simulation needed for this step.")
        # Return the *current* state without modification
        # Use deepcopy only if downstream might modify elements accidentally
        return image, copy.deepcopy(
            elements
        )  # Or just 'return image, elements' if mutation isn't a risk

    logger.debug(f"Simulating action: {plan.action} on element {plan.element_id}")
    new_image = image.copy()
    # IMPORTANT: Deep copy elements to avoid modifying previous steps' state
    new_elements = copy.deepcopy(elements)
    draw = ImageDraw.Draw(new_image)

    target_element = next((el for el in new_elements if el.id == plan.element_id), None)

    if not target_element:
        logger.warning(f"Simulation failed: Element ID {plan.element_id} not found.")
        return image, elements  # Return original state

    action = plan.action
    element_type = target_element.type

    try:
        # --- Simulate TYPE action ---
        if action == "type":
            if element_type == "text_field" and plan.text_to_type is not None:
                text_to_draw = plan.text_to_type
                target_element.content = text_to_draw  # Update element data
                abs_x, abs_y, abs_w, abs_h = _bounds_to_abs(target_element.bounds)

                # Mask password text for drawing
                if target_element.attributes.get("is_password"):
                    text_to_draw = "*" * len(text_to_draw)

                # Erase previous content by drawing background color
                draw.rectangle(
                    [(abs_x, abs_y), (abs_x + abs_w, abs_y + abs_h)],
                    fill="white",
                    outline="black",
                )
                # Draw new text (vertically centered)
                text_bbox = draw.textbbox((0, 0), text_to_draw, font=FONT)
                text_h = text_bbox[3] - text_bbox[1]
                draw.text(
                    (abs_x + 5, abs_y + (abs_h - text_h) / 2),
                    text_to_draw,
                    fill="black",
                    font=FONT,
                )
                logger.info(
                    f"Simulated typing '{plan.text_to_type}' into element {target_element.id}"
                )
                return new_image, new_elements
            else:
                logger.warning(
                    f"Cannot simulate 'type' on element type '{element_type}' or missing text."
                )
                return image, elements

        # --- Simulate CLICK action ---
        elif action == "click":
            # Click on Login Button
            if element_type == "button" and target_element.content == "Login":
                # Basic check: assume login succeeds if both fields have content
                username_filled = any(el.id == 0 and el.content for el in new_elements)
                password_filled = any(el.id == 1 and el.content for el in new_elements)
                if username_filled and password_filled:
                    logger.info("Simulating successful login transition.")
                    # Transition to logged-in screen
                    # Get username from element 0 content for personalization
                    login_username = next(
                        (el.content for el in new_elements if el.id == 0),
                        username_for_login,
                    )
                    return generate_logged_in_screen(
                        username=login_username
                    )  # Return new screen state
                else:
                    logger.warning(
                        "Simulating login click, but fields not filled. No state change."
                    )
                    # Optionally: Add an error message element to the current screen
                    return image, elements  # No state change if fields empty

            # Click on Checkbox
            elif element_type == "checkbox":
                is_checked = target_element.attributes.get("checked", False)
                target_element.attributes["checked"] = not is_checked  # Toggle state
                abs_x, abs_y, abs_w, abs_h = _bounds_to_abs(target_element.bounds)
                # Re-draw checkbox
                draw.rectangle(
                    [(abs_x, abs_y), (abs_x + abs_w, abs_y + abs_h)],
                    fill="white",
                    outline="black",
                )
                if not is_checked:  # Draw checkmark if it's now checked
                    draw.line(
                        [
                            (abs_x + 2, abs_y + abs_h // 2),
                            (abs_x + abs_w // 2, abs_y + abs_h - 2),
                        ],
                        fill="black",
                        width=2,
                    )
                    draw.line(
                        [
                            (abs_x + abs_w // 2, abs_y + abs_h - 2),
                            (abs_x + abs_w - 2, abs_y + 2),
                        ],
                        fill="black",
                        width=2,
                    )
                logger.info(
                    f"Simulated clicking checkbox {target_element.id}. New state: checked={not is_checked}"
                )
                return new_image, new_elements

            # Click on Link / Other Buttons (add more simulation logic here if needed)
            elif (
                element_type == "link" and target_element.content == "Forgot Password?"
            ):
                logger.info(
                    "Simulated clicking 'Forgot Password?' link. (No visual state change implemented)."
                )
                # Could transition to another screen if desired
                return image, elements  # No state change for now

            else:
                logger.warning(
                    f"Simulation for clicking element type '{element_type}' with content '{target_element.content}' not fully implemented."
                )
                return image, elements  # No change if click simulation not defined
        else:
            logger.warning(f"Action type '{action}' simulation not implemented.")
            return image, elements  # Return original state if action unknown

    except Exception as e:
        logger.error(f"Error during simulation: {e}", exc_info=True)
        return image, elements  # Return original state on error


# --- Visualization ---


def draw_highlight(
    image: Image.Image,
    element: UIElement,
    plan: Any,  # Add plan object (can use 'LLMActionPlan' if imported or from typing import Any)
    color: str = "lime",
    width: int = 3,
    dim_factor: float = 0.5,
    text_color: str = "black",  # Color for annotation text
    text_bg_color: tuple[int, int, int, int] = (
        255,
        255,
        255,
        200,
    ),  # Semi-transparent white bg for text
) -> Image.Image:
    """
    Draws highlight box, dims background, and adds text annotation for the planned action.

    Args:
        image: The source PIL Image.
        element: The UIElement to highlight.
        plan: The LLMActionPlan object containing the planned action details.
        color: The color of the highlight box.
        width: The line width of the highlight box.
        dim_factor: Factor to reduce brightness of non-highlighted areas.
        text_color: Color for the annotation text.
        text_bg_color: Background color for the annotation text.

    Returns:
        A new PIL Image with the effects.
    """
    if not element or not hasattr(element, "bounds") or not plan:
        logger.warning(
            "Attempted to draw highlight/text for invalid element or missing plan."
        )
        return image.copy()

    final_image = image.copy()

    try:
        abs_x, abs_y, abs_w, abs_h = _bounds_to_abs(element.bounds)
        element_box = (abs_x, abs_y, abs_x + abs_w, abs_y + abs_h)

        # --- Apply Dimming ---
        if 0.0 <= dim_factor < 1.0:
            enhancer = ImageEnhance.Brightness(final_image)
            dimmed_image = enhancer.enhance(dim_factor)
            crop_box = (
                max(0, element_box[0]),
                max(0, element_box[1]),
                min(image.width, element_box[2]),
                min(image.height, element_box[3]),
            )
            if crop_box[0] < crop_box[2] and crop_box[1] < crop_box[3]:
                original_element_area = image.crop(crop_box)
                dimmed_image.paste(original_element_area, (crop_box[0], crop_box[1]))
                final_image = dimmed_image
            else:
                logger.warning(
                    f"Invalid crop box {crop_box} for element {element.id}. Skipping paste."
                )
                final_image = dimmed_image

        # --- Draw Highlight Box ---
        draw = ImageDraw.Draw(final_image)
        draw.rectangle(
            [(element_box[0], element_box[1]), (element_box[2], element_box[3])],
            outline=color,
            width=width,
        )

        # --- Add Text Annotation ---
        try:
            # Construct text based on plan
            action_text = str(plan.action).capitalize()
            if plan.action == "type" and plan.text_to_type is not None:
                # Truncate long text for display
                text_preview = (
                    (plan.text_to_type[:20] + "...")
                    if len(plan.text_to_type) > 23
                    else plan.text_to_type
                )
                annotation_text = f"Next: {action_text} '{text_preview}'"
            else:
                annotation_text = f"Next: {action_text}"
                # Optionally add element content:
                # content_preview = (element.content[:15] + '...') if len(element.content) > 18 else element.content
                # if content_preview: annotation_text += f" '{content_preview}'"

            # Calculate text position (prefer placing above the box)
            margin = 5
            text_bbox = draw.textbbox((0, 0), annotation_text, font=FONT)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Center horizontally above box, clamp to image bounds
            text_x = max(
                margin,
                min(
                    abs_x + (abs_w - text_width) // 2,
                    image.width - text_width - margin,  # Ensure right edge fits
                ),
            )
            # Position above box, clamp to image bounds (top edge)
            text_y = max(margin, abs_y - text_height - margin)

            # Optional: Draw background rectangle for text readability
            bg_x0 = text_x - margin // 2
            bg_y0 = text_y - margin // 2
            bg_x1 = text_x + text_width + margin // 2
            bg_y1 = text_y + text_height + margin // 2
            # Ensure background rect is within image bounds
            bg_x0, bg_y0 = max(0, bg_x0), max(0, bg_y0)
            bg_x1, bg_y1 = min(final_image.width, bg_x1), min(final_image.height, bg_y1)
            if bg_x0 < bg_x1 and bg_y0 < bg_y1:  # Draw only if valid rect
                draw.rectangle([(bg_x0, bg_y0), (bg_x1, bg_y1)], fill=text_bg_color)

            # Draw the text
            draw.text((text_x, text_y), annotation_text, fill=text_color, font=FONT)

        except Exception as text_e:
            logger.warning(f"Failed to draw text annotation: {text_e}")
        # --- End Text Annotation ---

    except Exception as e:
        logger.error(
            f"Failed during drawing highlight/dimming/text for element {getattr(element, 'id', 'N/A')}: {e}",
            exc_info=True,
        )
        return image.copy()

    return final_image
